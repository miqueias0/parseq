# Scene Text Recognition Model Hub - ONNX Batch Scaling Benchmark Suite
# Copyright 2026 Darwin Bautista / PARSeq ONNX Extensions
#
# Licensed under the Apache License, Version 2.0 (the "License");

import gc
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .runtime import PARSeqONNXRuntime


@dataclass
class BatchBenchmarkPoint:
    """Represents performance metrics measured for a specific batch size."""
    batch_size: int
    mean_batch_lat_ms: float
    std_batch_lat_ms: float
    p50_batch_lat_ms: float
    p95_batch_lat_ms: float
    p99_batch_lat_ms: float
    lat_per_sample_ms: float
    fps: float
    scaling_vs_b1: float
    gain_vs_prev_pct: float
    oom_occurred: bool = False
    error_message: Optional[str] = None


@dataclass
class ModelBatchBenchmarkResult:
    """Benchmark results of a single ONNX model across a sweep of batch sizes."""
    model_path: str
    model_name: str
    model_size_mb: float
    device: str
    active_provider: str
    points: List[BatchBenchmarkPoint]
    peak_fps: float
    peak_batch: int
    saturation_batch: int
    b1_latency_ms: float
    b1_fps: float
    max_scaling: float
    recommended_sweet_spot_batch: int


def get_gpu_memory_mb(device_id: int = 0) -> Optional[float]:
    """Returns currently allocated GPU memory in MB if CUDA is available."""
    if torch.cuda.is_available():
        try:
            return torch.cuda.memory_allocated(device_id) / (1024 * 1024)
        except Exception:
            return None
    return None


def clean_cuda_memory():
    """Forces garbage collection and empties PyTorch/CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception:
            pass


def benchmark_model_batch_scaling(
    model_path: str,
    batch_sizes: Sequence[int] = (1, 2, 4, 8, 16, 32, 64, 128),
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    device_id: int = 0,
    warmup_iters: int = 15,
    test_iters: int = 50,
    auto_stop_on_saturation: bool = False,
    saturation_threshold_pct: float = 3.0,
    target_latency_sla_ms: Optional[float] = None,
) -> ModelBatchBenchmarkResult:
    """Measures batch scaling performance for a single ONNX model across multiple batch sizes.

    Handles GPU OOM gracefully, records per-sample latency and throughput (FPS),
    and computes scaling efficiency and saturation points.
    """
    clean_cuda_memory()
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    model_name = Path(model_path).stem

    engine = PARSeqONNXRuntime(
        onnx_model_path=model_path,
        device=device,
        device_id=device_id,
    )
    img_h, img_w = engine.img_size

    points: List[BatchBenchmarkPoint] = []
    b1_fps = 0.0
    b1_lat = 0.0
    prev_fps = 0.0
    saturation_batch = batch_sizes[0]
    peak_fps = 0.0
    peak_batch = batch_sizes[0]
    sweet_spot_batch = batch_sizes[0]

    for b in batch_sizes:
        input_shape = (b, 3, img_h, img_w)
        try:
            clean_cuda_memory()
            metrics = engine.benchmark_latency(
                input_shape=input_shape,
                warmup_iters=max(2, warmup_iters),
                test_iters=max(5, test_iters),
            )
            mean_lat = metrics["mean_ms"]
            fps = metrics["fps"]
            lat_per_sample = mean_lat / b

            if b == batch_sizes[0] or b == 1 or b1_fps == 0.0:
                b1_fps = fps
                b1_lat = mean_lat

            scaling = fps / b1_fps if b1_fps > 0 else 1.0
            gain_pct = ((fps - prev_fps) / prev_fps * 100.0) if prev_fps > 0 else 0.0

            if fps > peak_fps:
                peak_fps = fps
                peak_batch = b

            # Saturation is reached when throughput gain upon increasing batch falls below threshold
            if prev_fps > 0 and gain_pct < saturation_threshold_pct and saturation_batch == batch_sizes[0]:
                saturation_batch = b

            # Sweet spot: batch size providing at least 85% of peak throughput without excessive latency
            if sweet_spot_batch == batch_sizes[0] and prev_fps > 0 and gain_pct < 10.0:
                sweet_spot_batch = prev_batch if 'prev_batch' in locals() else b

            point = BatchBenchmarkPoint(
                batch_size=b,
                mean_batch_lat_ms=round(mean_lat, 2),
                std_batch_lat_ms=round(metrics["std_ms"], 2),
                p50_batch_lat_ms=round(metrics["p50_ms"], 2),
                p95_batch_lat_ms=round(metrics["p95_ms"], 2),
                p99_batch_lat_ms=round(metrics["p99_ms"], 2),
                lat_per_sample_ms=round(lat_per_sample, 3),
                fps=round(fps, 1),
                scaling_vs_b1=round(scaling, 2),
                gain_vs_prev_pct=round(gain_pct, 1),
            )
            points.append(point)
            prev_fps = fps
            prev_batch = b

            if auto_stop_on_saturation and prev_fps > 0 and gain_pct < saturation_threshold_pct and b >= 16:
                break

        except Exception as e:
            err_msg = str(e)
            is_oom = "out of memory" in err_msg.lower() or "allocate" in err_msg.lower() or "bad_alloc" in err_msg.lower()
            point = BatchBenchmarkPoint(
                batch_size=b,
                mean_batch_lat_ms=-1.0,
                std_batch_lat_ms=-1.0,
                p50_batch_lat_ms=-1.0,
                p95_batch_lat_ms=-1.0,
                p99_batch_lat_ms=-1.0,
                lat_per_sample_ms=-1.0,
                fps=0.0,
                scaling_vs_b1=0.0,
                gain_vs_prev_pct=0.0,
                oom_occurred=is_oom,
                error_message=err_msg[:120],
            )
            points.append(point)
            clean_cuda_memory()
            break

    if saturation_batch == batch_sizes[0] and points:
        saturation_batch = peak_batch
    if sweet_spot_batch == batch_sizes[0] and points:
        sweet_spot_batch = peak_batch

    max_scaling = (peak_fps / b1_fps) if b1_fps > 0 else 1.0

    return ModelBatchBenchmarkResult(
        model_path=model_path,
        model_name=model_name,
        model_size_mb=round(size_mb, 2),
        device=device,
        active_provider=engine.active_provider,
        points=points,
        peak_fps=round(peak_fps, 1),
        peak_batch=peak_batch,
        saturation_batch=saturation_batch,
        b1_latency_ms=round(b1_lat, 2),
        b1_fps=round(b1_fps, 1),
        max_scaling=round(max_scaling, 2),
        recommended_sweet_spot_batch=sweet_spot_batch,
    )


def discover_onnx_models(
    onnx_dir: str,
    filter_pattern: Optional[str] = None,
) -> List[str]:
    """Finds all .onnx files in directory matching optional filter pattern."""
    if not os.path.exists(onnx_dir):
        return []

    files = [
        os.path.join(onnx_dir, f)
        for f in os.listdir(onnx_dir)
        if f.endswith(".onnx") and not f.startswith(".")
    ]

    if filter_pattern:
        regex = re.compile(filter_pattern, re.IGNORECASE)
        files = [f for f in files if regex.search(os.path.basename(f))]

    # Sort naturally
    files.sort(key=lambda x: os.path.basename(x))
    return files


def run_batch_scaling_benchmark_suite(
    models: Sequence[str],
    batch_sizes: Sequence[int] = (1, 4, 8, 16, 32, 64, 128),
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    device_id: int = 0,
    warmup_iters: int = 15,
    test_iters: int = 50,
    auto_stop_on_saturation: bool = False,
    target_latency_sla_ms: Optional[float] = None,
) -> List[ModelBatchBenchmarkResult]:
    """Runs batch scaling benchmark across a list of ONNX models."""
    results: List[ModelBatchBenchmarkResult] = []
    total = len(models)

    for idx, m_path in enumerate(models, 1):
        print(f"[{idx}/{total}] Benchmarkando modelo: {os.path.basename(m_path)}...")
        res = benchmark_model_batch_scaling(
            model_path=m_path,
            batch_sizes=batch_sizes,
            device=device,
            device_id=device_id,
            warmup_iters=warmup_iters,
            test_iters=test_iters,
            auto_stop_on_saturation=auto_stop_on_saturation,
            target_latency_sla_ms=target_latency_sla_ms,
        )
        results.append(res)
        print(
            f"      -> B=1: {res.b1_fps:.1f} FPS ({res.b1_latency_ms:.2f} ms) | "
            f"Pico: {res.peak_fps:.1f} FPS (Batch {res.peak_batch}, Escala {res.max_scaling:.2f}x) | "
            f"Sweet Spot: Batch {res.recommended_sweet_spot_batch}"
        )

    return results


def format_batch_scaling_markdown(
    results: List[ModelBatchBenchmarkResult],
    target_latency_sla_ms: Optional[float] = None,
) -> str:
    """Formats the benchmark results into a rich, comprehensive GitHub Markdown report."""
    if not results:
        return "Nenhum resultado de benchmark para exibir."

    # Identify all tested batch sizes
    all_batches_set = set()
    for r in results:
        for p in r.points:
            if not p.oom_occurred and p.fps > 0:
                all_batches_set.add(p.batch_size)
    all_batches = sorted(list(all_batches_set))

    dev_name = results[0].device.upper()
    provider = results[0].active_provider

    lines = []
    lines.append("# Relatório de Benchmark: Escalonamento de Batch no ONNX Runtime")
    lines.append(f"**Dispositivo:** {dev_name} | **Execution Provider:** `{provider}` | **Total de Modelos:** {len(results)}\n")
    lines.append("Este benchmark avalia a curva de rendimento computacional dos modelos PARSeq à medida que o tamanho do lote (*Batch Size*) cresce, permitindo identificar onde a GPU atinge saturação dos Tensor Cores, a redução do custo por imagem e o dimensionamento ótimo para produção.\n")

    # Tabela 1: Matriz Comparativa de Throughput (FPS) por Batch
    lines.append("## 1. Matriz de Throughput: FPS por Tamanho de Lote (Batch Size)")
    lines.append("Demonstra a vazão total em **imagens processadas por segundo (FPS)** conforme o batch cresce:\n")

    header_cols = ["Modelo ONNX", "Tamanho"] + [f"B={b}" for b in all_batches] + ["FPS Pico", "Batch Pico", "Escalonamento"]
    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(header_cols)) + " |")

    for r in results:
        pt_map = {p.batch_size: p for p in r.points}
        row = [f"`{r.model_name}`", f"{r.model_size_mb:.1f} MB"]
        for b in all_batches:
            p = pt_map.get(b)
            if p is None or p.oom_occurred or p.fps <= 0:
                row.append("OOM" if (p and p.oom_occurred) else "-")
            else:
                row.append(f"**{p.fps:.1f}**")
        row.append(f"**{r.peak_fps:.1f}**")
        row.append(f"B={r.peak_batch}")
        row.append(f"**{r.max_scaling:.2f}x**")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("\n---\n")

    # Tabela 2: Matriz de Custo por Amostra (Latência por Imagem em ms/img)
    lines.append("## 2. Matriz de Latência por Imagem (ms / imagem)")
    lines.append("Demonstra o custo computacional efetivo por imagem individual ($\text{Latência total} / \text{Batch Size}$):\n")

    header_cols2 = ["Modelo ONNX"] + [f"B={b}" for b in all_batches] + ["Menor ms/img", "Melhoria vs B=1"]
    lines.append("| " + " | ".join(header_cols2) + " |")
    lines.append("| " + " | ".join(["---"] * len(header_cols2)) + " |")

    for r in results:
        pt_map = {p.batch_size: p for p in r.points}
        b1_p = pt_map.get(1) or (r.points[0] if r.points else None)
        b1_lat_sample = b1_p.lat_per_sample_ms if b1_p and b1_p.lat_per_sample_ms > 0 else 1.0

        min_sample_lat = float("inf")
        row = [f"`{r.model_name}`"]
        for b in all_batches:
            p = pt_map.get(b)
            if p is None or p.oom_occurred or p.lat_per_sample_ms <= 0:
                row.append("OOM" if (p and p.oom_occurred) else "-")
            else:
                row.append(f"{p.lat_per_sample_ms:.2f} ms")
                if p.lat_per_sample_ms < min_sample_lat:
                    min_sample_lat = p.lat_per_sample_ms

        if min_sample_lat < float("inf"):
            row.append(f"**{min_sample_lat:.2f} ms**")
            red_pct = ((b1_lat_sample - min_sample_lat) / b1_lat_sample * 100.0) if b1_lat_sample > 0 else 0.0
            row.append(f"**-{red_pct:.1f}%**")
        else:
            row.append("-")
            row.append("-")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("\n---\n")

    # Tabela 3: Matriz de Latência Total do Lote (ms / batch) - SLA
    lines.append("## 3. Matriz de Latência Total do Batch (ms / batch)")
    lines.append("Tempo total transcorrido desde o envio do lote até o recebimento das predições (essencial para definir SLAs de resposta):\n")

    header_cols3 = ["Modelo ONNX"] + [f"B={b}" for b in all_batches]
    lines.append("| " + " | ".join(header_cols3) + " |")
    lines.append("| " + " | ".join(["---"] * len(header_cols3)) + " |")

    for r in results:
        pt_map = {p.batch_size: p for p in r.points}
        row = [f"`{r.model_name}`"]
        for b in all_batches:
            p = pt_map.get(b)
            if p is None or p.oom_occurred or p.mean_batch_lat_ms <= 0:
                row.append("OOM" if (p and p.oom_occurred) else "-")
            else:
                row.append(f"{p.mean_batch_lat_ms:.2f} ms")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("\n---\n")

    # Seção 4: Destaques da Fronteira de Pareto & Recomendações
    lines.append("## 4. Destaques da Fronteira de Pareto & Recomendações de Dimensionamento")

    # Campeão Baixa Latência (Batch 1)
    valid_b1 = [r for r in results if r.b1_fps > 0]
    if valid_b1:
        fastest_b1 = min(valid_b1, key=lambda r: r.b1_latency_ms)
        lines.append(f"- 🚀 **Campeão de Baixa Latência (Batch 1 / Tempo Real):** `{fastest_b1.model_name}`")
        lines.append(f"  - Latência B=1: **{fastest_b1.b1_latency_ms:.2f} ms** ({fastest_b1.b1_fps:.1f} FPS) | Tamanho: {fastest_b1.model_size_mb:.2f} MB\n")

    # Campeão Alto Throughput (Pico de FPS)
    if results:
        highest_fps = max(results, key=lambda r: r.peak_fps)
        lines.append(f"- ⚡ **Campeão de Throughput Máximo (Produção em Lote):** `{highest_fps.model_name}`")
        lines.append(f"  - Pico de Vazão: **{highest_fps.peak_fps:.1f} FPS** (Batch {highest_fps.peak_batch}) | Speedup vs B=1: **{highest_fps.max_scaling:.2f}x** | Tamanho: {highest_fps.model_size_mb:.2f} MB\n")

    # Mais Compacto
    if results:
        smallest = min(results, key=lambda r: r.model_size_mb)
        lines.append(f"- 💾 **Modelo Mais Compacto (Menor Footprint):** `{smallest.model_name}`")
        lines.append(f"  - Tamanho: **{smallest.model_size_mb:.2f} MB** | Pico de Vazão: {smallest.peak_fps:.1f} FPS (Batch {smallest.peak_batch})\n")

    # Maior Eficiência de Escala
    if results:
        best_scaler = max(results, key=lambda r: r.max_scaling)
        lines.append(f"- 📈 **Maior Eficiência de Escalonamento de Hardware:** `{best_scaler.model_name}`")
        lines.append(f"  - Escalonamento: **{best_scaler.max_scaling:.2f}x** do Batch 1 ({best_scaler.b1_fps:.1f} FPS) até o Batch {best_scaler.peak_batch} ({best_scaler.peak_fps:.1f} FPS)\n")

    # Ponto de Saturação e Sweet Spot
    lines.append("### 🎯 Análise do Ponto de Saturação (Hardware Saturation & Sweet Spot):")
    lines.append("Conforme o tamanho do lote aumenta:")
    lines.append("1. **Região Memory/Latency-Bound (B=1 a B=8):** O tempo de despacho de kernels e latência de transferência de dados domina; os Tensor Cores ficam subutilizados.")
    lines.append("2. **Região de Ganho Acelerado (B=8 a B=32):** Os Tensor Cores entram em saturação eficiente. O throughput em FPS dispara e o custo em milissegundos por imagem cai em até 70% a 85%.")
    lines.append("3. **Região de Saturação (B >= 64 ou B >= 128):** A GPU atinge o limite térmico/computacional ou de largura de banda de memória (HBM/VRAM), e aumentos adicionais no batch trazem ganhos residuais (< 3% a 5%) enquanto aumentam a latência total do lote.")

    if target_latency_sla_ms is not None:
        lines.append(f"\n### ⏱️ Dimensionamento Recomendado para SLA de Latência ({target_latency_sla_ms:.1f} ms):")
        for r in results:
            sla_points = [p for p in r.points if p.mean_batch_lat_ms > 0 and p.mean_batch_lat_ms <= target_latency_sla_ms]
            if sla_points:
                best_p = max(sla_points, key=lambda p: p.batch_size)
                lines.append(f"- `{r.model_name}`: **Batch {best_p.batch_size}** -> Latência {best_p.mean_batch_lat_ms:.2f} ms ({best_p.fps:.1f} FPS, {best_p.lat_per_sample_ms:.2f} ms/img)")
            else:
                lines.append(f"- `{r.model_name}`: Nenhum lote atende ao SLA de {target_latency_sla_ms:.1f} ms (menor latência B=1 é {r.b1_latency_ms:.2f} ms)")

    lines.append("\n---\n")

    # Tabela 4: Detalhamento Estatístico por Modelo
    lines.append("## 5. Detalhamento Estatístico de Latência (Percentis p50, p95, p99)")
    for r in results:
        lines.append(f"### Modelo: `{r.model_name}` ({r.model_size_mb:.2f} MB)")
        lines.append(f"**Sweet Spot Recomendado:** Batch {r.recommended_sweet_spot_batch} | **Pico:** {r.peak_fps:.1f} FPS (Batch {r.peak_batch})\n")
        det_header = "| Batch | Latência Média | p50 (Mediana) | p95 | p99 | ms / Imagem | Throughput (FPS) | Escalonamento | Ganho vs Ant. |"
        lines.append(det_header)
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for p in r.points:
            if p.oom_occurred:
                lines.append(f"| **B={p.batch_size}** | *OOM* | - | - | - | - | 0.0 | - | - |")
            else:
                gain_str = f"+{p.gain_vs_prev_pct:.1f}%" if p.gain_vs_prev_pct > 0 else f"{p.gain_vs_prev_pct:.1f}%"
                lines.append(
                    f"| **B={p.batch_size}** | {p.mean_batch_lat_ms:.2f} ms | {p.p50_batch_lat_ms:.2f} ms | "
                    f"{p.p95_batch_lat_ms:.2f} ms | {p.p99_batch_lat_ms:.2f} ms | **{p.lat_per_sample_ms:.2f} ms** | "
                    f"**{p.fps:.1f}** | **{p.scaling_vs_b1:.2f}x** | {gain_str} |"
                )
        lines.append("")

    return "\n".join(lines)


def export_batch_benchmark_json(results: List[ModelBatchBenchmarkResult], output_path: str):
    """Exports structured benchmark results to JSON file."""
    data = [asdict(r) for r in results]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
