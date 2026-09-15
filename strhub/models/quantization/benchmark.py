# Scene Text Recognition Model Hub - Quantization Benchmark Engine
# Rigorous measurement of Latency (P50, P90, P99), Throughput (FPS), Memory, and Accuracy

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from nltk import edit_distance


@dataclass
class BenchmarkMetrics:
    name: str
    device: str
    batch_size: int
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p90_ms: float
    latency_p99_ms: float
    fps: float
    memory_mb: float
    accuracy_pct: Optional[float] = None
    cer_pct: Optional[float] = None
    ned_pct: Optional[float] = None
    speedup_vs_baseline: float = 1.0


class BenchmarkEngine:
    """Benchmark engine adhering to DL and hardware benchmarking protocols."""

    @staticmethod
    def measure_model_memory_mb(model: nn.Module) -> float:
        """Calculates physical memory consumption of model weights and buffers in MB."""
        total_bytes = 0
        for p in model.parameters():
            total_bytes += p.numel() * p.element_size()
        for b in model.buffers():
            total_bytes += b.numel() * b.element_size()
        return total_bytes / (1024.0 * 1024.0)

    @staticmethod
    @torch.inference_mode()
    def measure_latency_and_fps(
        model: Any,
        device: str = "cpu",
        batch_size: int = 1,
        iterations: int = 100,
        warmup: int = 15,
        img_size: Tuple[int, int] = (32, 128),
        is_onnx: bool = False,
        onnx_session: Any = None,
    ) -> Dict[str, float]:
        """Profiles execution latency and throughput with strict CUDA synchronization and percentiles."""
        is_cuda = (device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda"))
        if is_cuda and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        dummy_tensor = torch.randn(batch_size, 3, img_size[0], img_size[1])
        if not is_onnx and hasattr(model, "to"):
            model = model.eval().to(device)
            dummy_tensor = dummy_tensor.to(device)

        dummy_numpy = dummy_tensor.cpu().numpy()

        # 1. Warm-up
        for _ in range(warmup):
            if is_onnx:
                _ = onnx_session.run(None, {"image": dummy_numpy})
            else:
                _ = model(dummy_tensor)
            if is_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()

        # 2. Timing Loop
        latencies_ms: List[float] = []
        for _ in range(iterations):
            if is_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            if is_onnx:
                _ = onnx_session.run(None, {"image": dummy_numpy})
            else:
                _ = model(dummy_tensor)

            if is_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

        latencies_arr = np.array(latencies_ms)
        mean_lat = float(np.mean(latencies_arr))
        p50 = float(np.percentile(latencies_arr, 50))
        p90 = float(np.percentile(latencies_arr, 90))
        p99 = float(np.percentile(latencies_arr, 99))
        fps = (batch_size * 1000.0) / mean_lat if mean_lat > 0 else 0.0

        return {
            "mean_ms": mean_lat,
            "p50_ms": p50,
            "p90_ms": p90,
            "p99_ms": p99,
            "fps": fps,
        }

    @staticmethod
    @torch.inference_mode()
    def evaluate_model_accuracy(
        model: nn.Module,
        dataloader: Any,
        device: str = "cpu",
        max_samples: Optional[int] = None,
    ) -> Dict[str, float]:
        """Computes Word Accuracy, Character Error Rate (CER), and Normalized Edit Distance (NED)."""
        model = model.eval().to(device)
        total = 0
        correct = 0
        total_ned = 0.0
        total_char_dist = 0
        total_char_len = 0

        # Extract tokenizer
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None and hasattr(model, "model"):
            tokenizer = getattr(model.model, "tokenizer", None)

        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                images, labels = batch[0], batch[1]
            else:
                continue

            images = images.to(device)
            logits = model(images)
            probs = logits.softmax(-1)
            
            # Greedy decode
            preds, _ = tokenizer.decode(probs)

            for pred, label in zip(preds, labels):
                # Clean EOS / BOS tokens
                pred = pred.split("[EOS]")[0].strip()
                label = label.strip()

                is_correct = (pred == label)
                if is_correct:
                    correct += 1

                ed = edit_distance(pred, label)
                max_len = max(len(pred), len(label))
                ned = 1.0 - (ed / max_len) if max_len > 0 else 1.0
                total_ned += ned

                total_char_dist += ed
                total_char_len += len(label)
                total += 1

                if max_samples and total >= max_samples:
                    break

            if max_samples and total >= max_samples:
                break

        acc = (correct / total * 100.0) if total > 0 else 0.0
        ned_pct = (total_ned / total * 100.0) if total > 0 else 0.0
        cer = (total_char_dist / max(1, total_char_len) * 100.0)

        return {
            "accuracy": acc,
            "ned": ned_pct,
            "cer": cer,
            "samples": total,
        }

    @staticmethod
    def format_results_table(results: List[BenchmarkMetrics]) -> str:
        """Generates a GitHub-flavored markdown comparison table."""
        lines = [
            "| Modelo / Configuração | Dispositivo | Latência P50 (ms) | Média (ms) | Throughput (FPS) | Memória (MB) | Acurácia (%) | CER (%) | Speedup |",
            "|:---|:---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        baseline_lat = results[0].latency_mean_ms if results else 1.0
        for r in results:
            speedup = f"{(baseline_lat / max(r.latency_mean_ms, 1e-6)):.2f}x"
            acc_str = f"{r.accuracy_pct:.2f}%" if r.accuracy_pct is not None else "N/A"
            cer_str = f"{r.cer_pct:.2f}%" if r.cer_pct is not None else "N/A"
            lines.append(
                f"| **{r.name}** | {r.device} | {r.latency_p50_ms:.2f} ms | {r.latency_mean_ms:.2f} ms | "
                f"{r.fps:.1f} FPS | {r.memory_mb:.2f} MB | {acc_str} | {cer_str} | **{speedup}** |"
            )
        return "\n".join(lines)
