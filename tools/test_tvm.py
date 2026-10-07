#!/usr/bin/env python3
"""High-Performance Apache TVM Evaluator for PARSeq / STRHub.
Extracts throughput and latency from compiled .so / .tar models using batched execution
via Apache TVM runtime, measuring:
- Exact Plate / Sequence Accuracy (%)
- 1 - Normalized Edit Distance (1 - NED %)
- Sequence Confidence (%)
- Average Label Length
- Batch & Per-Sample Latency (ms)
- Inference Throughput (FPS)
Mirrors exact architectural parity and CLI interface of tools/test_tensorrt.py.
"""

import os
import sys

# Ensure UTF-8 output encoding across platforms
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

os.environ["TVM_FFI_DISABLE_TORCH_C_DLPACK"] = "1"

import time
import string
import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
from nltk import edit_distance
import numpy as np
import torch

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args
from strhub.models.tvm_utils import TVMRuntimeSession, get_tvm_device


@dataclass
class Result:
    dataset: str
    num_samples: int = 0
    accuracy: float = 0.0
    ned: float = 0.0
    confidence: float = 0.0
    label_length: float = 0.0
    latency_ms: float = 0.0
    fps: float = 0.0

    def merge(self, result):
        self.dataset = result.dataset
        self.accuracy += result.accuracy
        self.num_samples = result.num_samples
        self.ned += result.ned
        self.confidence += result.confidence
        self.label_length += result.label_length
        self.latency_ms += result.latency_ms
        self.fps += result.fps

    def final_result(self, iterations):
        if iterations > 0:
            self.accuracy /= iterations
            self.ned /= iterations
            self.confidence /= iterations
            self.label_length /= iterations
            self.latency_ms /= iterations
            self.fps = 1000.0 / self.latency_ms if self.latency_ms > 0 else 0.0


def print_results_table(results: List[Result], file=None):
    if not results:
        print("Nenhum resultado para exibir.", file=file)
        return
    w = max(map(len, [r.dataset for r in results]))
    w = max(w, len('Dataset'), len('Combined'))

    print(
        '| {:<{w}} | # samples | Accuracy | 1 - NED | Confidence | Label Length | Latency (ms) | Throughput (FPS) |'.format(
            'Dataset', w=w), file=file)
    print(
        '|:{:-<{w}}:|----------:|---------:|--------:|-----------:|-------------:|-------------:|-----------------:|'.format(
            '----', w=w), file=file)

    c = Result('Combined', 0, 0, 0, 0, 0, 0, 0)
    total_time_combined = 0.0

    for res in results:
        c.num_samples += res.num_samples
        c.accuracy += res.num_samples * res.accuracy
        c.ned += res.num_samples * res.ned
        c.confidence += res.num_samples * res.confidence
        c.label_length += res.num_samples * res.label_length
        c.latency_ms += res.num_samples * res.latency_ms
        if res.fps > 0:
            total_time_combined += res.num_samples / res.fps

        print(
            f'| {res.dataset:<{w}} | {res.num_samples:>9} | {res.accuracy:>8.2f} | {res.ned:>7.2f} '
            f'| {res.confidence:>10.2f} | {res.label_length:>12.2f} | {res.latency_ms:>12.2f} | {res.fps:>16.1f} |',
            file=file,
        )

    if c.num_samples > 0:
        c.accuracy /= c.num_samples
        c.ned /= c.num_samples
        c.confidence /= c.num_samples
        c.label_length /= c.num_samples
        c.latency_ms /= c.num_samples
        c.fps = c.num_samples / total_time_combined if total_time_combined > 0 else 0.0

    print(
        '|-{:-<{w}}-|-----------|----------|---------|------------|--------------|--------------|------------------|'.format(
            '----', w=w), file=file)
    print(
        f'| {c.dataset:<{w}} | {c.num_samples:>9} | {c.accuracy:>8.2f} | {c.ned:>7.2f} '
        f'| {c.confidence:>10.2f} | {c.label_length:>12.2f} | {c.latency_ms:>12.2f} | {c.fps:>16.1f} |',
        file=file,
    )

    return c


class FastTVMEvaluator:
    """High-throughput batched evaluator directly utilizing Apache TVM execution runtime."""

    def __init__(
        self,
        lib_path: str,
        base_system,
        device: str = "cuda",
    ):
        if not os.path.isfile(lib_path):
            raise FileNotFoundError(f"Arquivo TVM compilado (.so / .tar) não encontrado em: {lib_path}")

        self.tokenizer = base_system.tokenizer
        self.charset_adapter = base_system.charset_adapter
        self.hparams = base_system.hparams
        self.lib_path = lib_path

        dev_str = device.lower()
        if "cuda" in dev_str and not torch.cuda.is_available():
            dev_str = "cpu"
        self.device = torch.device(dev_str)

        print(f"Carregando Apache TVM Library: {lib_path} no dispositivo: {dev_str}")
        self.session = TVMRuntimeSession(lib_path, device=dev_str)
        self.img_h = self.hparams.img_size[0]
        self.img_w = self.hparams.img_size[1]

    def warmup(self, batch_size: int = 1, iters: int = 10) -> None:
        """Aquece o runtime TVM antes das medições estatísticas."""
        dummy = np.random.randn(max(1, batch_size), 3, self.img_h, self.img_w).astype(np.float32)
        for _ in range(iters):
            try:
                _ = self.session.run(dummy)
            except Exception:
                for b in range(batch_size):
                    _ = self.session.run(dummy[b:b+1])
        if "cuda" in str(self.device) and torch.cuda.is_available():
            torch.cuda.synchronize()

    def evaluate_batch(self, images: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Executa inferência em lote no Apache TVM e calcula o tempo de execução em milissegundos."""
        bs = images.shape[0]
        imgs_np = images.detach().cpu().numpy()

        t_start = time.perf_counter()
        try:
            out_np = self.session.run(imgs_np)
            batch_logits = torch.from_numpy(out_np)
        except Exception:
            # Fallback a fatiamento caso a biblioteca seja compilada com batch fixo
            logits_list = []
            for i in range(bs):
                single_out = self.session.run(imgs_np[i:i+1])
                logits_list.append(torch.from_numpy(single_out))
            batch_logits = torch.cat(logits_list, dim=0)

        if "cuda" in str(self.device) and torch.cuda.is_available():
            torch.cuda.synchronize()
        t_end = time.perf_counter()

        elapsed_ms = (t_end - t_start) * 1000.0
        return batch_logits, elapsed_ms


@torch.inference_mode()
def run(
    evaluator: FastTVMEvaluator,
    dataloaders: Dict[str, Any],
    max_samples: Optional[int] = None,
    iterations: int = 1
) -> List[Result]:
    eval_results: List[Result] = []

    # Warmup
    evaluator.warmup(batch_size=1, iters=5)

    for it in range(iterations):
        for name, dataloader in dataloaders.items():
            total = 0
            correct = 0
            ned = 0
            confidence = 0.0
            label_length = 0
            total_time_ms = 0.0

            pbar = tqdm(dataloader, desc=f"{name} (iter {it+1}/{iterations})", leave=False)
            for batch_idx, (images, labels) in enumerate(pbar):
                if max_samples and total >= max_samples:
                    break

                batch_logits, batch_time_ms = evaluator.evaluate_batch(images)
                total_time_ms += batch_time_ms

                probs = batch_logits.softmax(-1)
                preds, prob_tuples = evaluator.tokenizer.decode(probs)

                for pred, prob, gt in zip(preds, prob_tuples, labels):
                    confidence += prob.prod().item()
                    pred = evaluator.charset_adapter(pred)
                    ned += edit_distance(pred, gt) / max(len(pred), len(gt), 1)
                    if pred == gt:
                        correct += 1
                    total += 1
                    label_length += len(pred)

                    if max_samples and total >= max_samples:
                        break

            if total == 0:
                continue

            acc_pct = 100.0 * (correct / total)
            ned_pct = 100.0 * (1.0 - ned / total)
            conf_pct = 100.0 * (confidence / total)
            avg_len = label_length / total
            latency_ms = total_time_ms / total
            fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0

            iter_result = Result(
                dataset=name,
                num_samples=total,
                accuracy=acc_pct,
                ned=ned_pct,
                confidence=conf_pct,
                label_length=avg_len,
                latency_ms=latency_ms,
                fps=fps,
            )

            if it == 0:
                eval_results.append(iter_result)
            else:
                eval_results[list(dataloaders.keys()).index(name)].merge(iter_result)

    for r in eval_results:
        r.final_result(iterations)

    return eval_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Apache TVM compiled libraries on text recognition benchmarks")
    parser.add_argument("lib", nargs="?", default=None, help="Caminho da biblioteca TVM (.so / .tar)")
    parser.add_argument("--lib_path", default=None, help="Caminho alternativo para a biblioteca TVM")
    parser.add_argument("--base_checkpoint", default="pretrained/parseq_alpr_98.5.ckpt",
                        help="Checkpoint PyTorch base para carregamento de tokenizer e hparams")
    parser.add_argument("--data_root", default="data", help="Diretório raiz dos datasets LMDB")
    parser.add_argument("--batch_size", type=int, default=1, help="Tamanho do lote")
    parser.add_argument("--num_workers", type=int, default=0, help="Workers do DataLoader")
    parser.add_argument("--cased", action="store_true", default=False, help="Avaliação cased")
    parser.add_argument("--punctuation", action="store_true", default=False, help="Avaliação com pontuação")
    parser.add_argument("--new", action="store_true", default=False, help="Usar benchmark estendido")
    parser.add_argument("--rotation", type=int, default=0, help="Rotação dos frames")
    parser.add_argument("--device", default="cuda", help="cuda ou cpu")
    parser.add_argument("--datasets", nargs="+", default=None, help="Datasets específicos para teste")
    parser.add_argument("--max_samples", type=int, default=None, help="Limite máximo de amostras")
    parser.add_argument("--iterations", type=int, default=1, help="Número de repetições de teste")
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)

    lib_path = args.lib or args.lib_path
    if not lib_path:
        parser.error("Caminho da biblioteca TVM é obrigatório.")

    charset_test = string.digits + string.ascii_lowercase
    if args.cased:
        charset_test += string.ascii_uppercase
    if args.punctuation:
        charset_test += string.punctuation
    kwargs.update({"charset_test": charset_test})

    base_sys = load_from_checkpoint(args.base_checkpoint, **kwargs).eval()
    evaluator = FastTVMEvaluator(lib_path, base_sys, device=args.device)

    dm = SceneTextDataModule(
        args.data_root,
        "_unused_",
        args.batch_size,
        args.num_workers,
        False,
        args.rotation,
        base_sys.hparams.img_size,
        base_sys.hparams.max_label_length,
        charset_test=charset_test
    )

    datasets = args.datasets or (SceneTextDataModule.BENCHMARK_SUB if not args.new else SceneTextDataModule.BENCHMARK)
    test_loaders = dm.test_dataloaders(datasets)

    print(f"\nIniciando Avaliação Apache TVM em: {datasets}...")
    results = run(evaluator, test_loaders, max_samples=args.max_samples, iterations=args.iterations)

    print("\n=== Tabela de Resultados TVM ===")
    print_results_table(results)


if __name__ == "__main__":
    main()
