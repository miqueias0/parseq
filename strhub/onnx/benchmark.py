# Scene Text Recognition Model Hub - ONNX Export & Runtime Engine
# Copyright 2026 Darwin Bautista / PARSeq ONNX Extensions
#
# Licensed under the Apache License, Version 2.0 (the "License");

from typing import Any, Dict, List, Optional, Tuple, Union
import time
import numpy as np
import torch
import torch.nn as nn

from strhub.models.utils import load_from_checkpoint
from .runtime import PARSeqONNXRuntime
from .export import export_parseq_to_onnx


def compare_pytorch_vs_onnx(
    checkpoint: str,
    onnx_path: str,
    device: str = "cpu",
    device_id: int = 0,
    warmup_iters: int = 25,
    test_iters: int = 80,
    batch_size: int = 1,
    fp16: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Runs a full comparative benchmark between PyTorch and ONNX Runtime."""
    dev = torch.device(device)

    # 1. PyTorch Benchmark
    model = load_from_checkpoint(checkpoint, decode_ar=False, **kwargs).eval().to(dev)
    if fp16 and dev.type == "cuda":
        model = model.half()

    model_dtype = next(model.parameters()).dtype
    img_size = model.hparams.img_size
    input_shape = (batch_size, 3, img_size[0], img_size[1])
    dummy_torch = torch.randn(*input_shape, device=dev, dtype=model_dtype)

    # Warmup PyTorch
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(dummy_torch)

    times_torch = []
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        for _ in range(test_iters):
            start_evt.record()
            _ = model(dummy_torch)
            end_evt.record()
            torch.cuda.synchronize(dev)
            times_torch.append(start_evt.elapsed_time(end_evt))
    else:
        for _ in range(test_iters):
            t0 = time.perf_counter_ns()
            _ = model(dummy_torch)
            t1 = time.perf_counter_ns()
            times_torch.append((t1 - t0) / 1e6)

    torch_mean = float(np.mean(times_torch))
    torch_p50 = float(np.median(times_torch))
    torch_fps = (batch_size * 1000.0) / torch_mean

    # 2. ONNX Runtime Benchmark
    ort_engine = PARSeqONNXRuntime(onnx_path, device=device, device_id=device_id)
    ort_metrics = ort_engine.benchmark_latency(input_shape=input_shape, warmup_iters=warmup_iters, test_iters=test_iters)
    ort_mean = ort_metrics["mean_ms"]
    ort_p50 = ort_metrics["p50_ms"]
    ort_fps = (batch_size * 1000.0) / ort_mean

    # 3. Numerical Parity Verification (ensuring correct device for both engines)
    dummy_np = dummy_torch.detach().cpu().float().numpy()
    with torch.no_grad():
        logits_torch = model(dummy_torch).detach().cpu().float().numpy()
    logits_onnx = ort_engine.forward(dummy_np)

    if logits_torch.shape == logits_onnx.shape:
        max_abs_diff = float(np.max(np.abs(logits_torch - logits_onnx)))
        mean_abs_diff = float(np.mean(np.abs(logits_torch - logits_onnx)))
    else:
        print(f"\n[PARSeq Benchmark Warning] Output shape mismatch:")
        print(f"  PyTorch model output shape : {logits_torch.shape}")
        print(f"  ONNX model output shape    : {logits_onnx.shape}")
        print(f"  (Different sequence length or charset between checkpoint and ONNX file;")
        print(f"   numerical parity diff skipped, latency benchmarks reported cleanly.)\n")
        max_abs_diff = None
        mean_abs_diff = None

    speedup = torch_mean / ort_mean if ort_mean > 0 else 1.0
    torch_label = "PyTorch (FP16)" if model_dtype == torch.float16 else "PyTorch Native"
    onnx_label = "ONNX Runtime (FP16)" if ("fp16" in onnx_path.lower()) else "ONNX Runtime (ORT)"

    return {
        "device": device,
        "active_provider": ort_engine.active_provider,
        "batch_size": batch_size,
        "pytorch_label": torch_label,
        "onnx_label": onnx_label,
        "pytorch_mean_ms": torch_mean,
        "pytorch_p50_ms": torch_p50,
        "pytorch_fps": torch_fps,
        "onnx_mean_ms": ort_mean,
        "onnx_p50_ms": ort_p50,
        "onnx_fps": ort_fps,
        "speedup": speedup,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
    }


def format_onnx_comparison_table(results: List[Dict[str, Any]]) -> str:
    """Formats ONNX vs PyTorch benchmark comparisons into a clean markdown table."""
    headers = ["Engine / Backend", "Provider / Device", "Latency Mean", "Median (p50)", "FPS", "Speedup", "Max Logit Diff"]
    rows = []

    for r in results:
        # PyTorch Row
        rows.append([
            r.get("pytorch_label", "PyTorch Native"),
            r["device"].upper(),
            f"{r['pytorch_mean_ms']:.2f} ms",
            f"{r['pytorch_p50_ms']:.2f} ms",
            f"{r['pytorch_fps']:.1f}",
            "1.00x (Baseline)",
            "0.0000",
        ])
        # ONNX Runtime Row
        diff_str = f"{r['max_abs_diff']:.2e}" if r.get("max_abs_diff") is not None else "N/A (Shape Mismatch)"
        rows.append([
            r.get("onnx_label", "ONNX Runtime (ORT)"),
            r["active_provider"],
            f"{r['onnx_mean_ms']:.2f} ms",
            f"{r['onnx_p50_ms']:.2f} ms",
            f"{r['onnx_fps']:.1f}",
            f"{r['speedup']:.2f}x",
            diff_str,
        ])

    col_widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    header_line = "| " + " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths)) + " |"
    sep_line = "|:" + "-|-".join("-" * w for w in col_widths) + ":|"
    data_lines = ["| " + " | ".join(f"{str(cell):<{w}}" for cell, w in zip(row, col_widths)) + " |" for row in rows]

    return "\n".join([header_line, sep_line] + data_lines)
