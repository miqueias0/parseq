import os
import sys
import time
import json
import argparse
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from strhub.models.utils import load_from_checkpoint
from strhub.models.parseq.quantized_parseq import create_model_variant


def benchmark_model(
    model: torch.nn.Module,
    tokenizer,
    batch_size: int = 1,
    img_size: tuple = (32, 128),
    num_warmup: int = 50,
    num_iterations: int = 200,
    device: str = "cuda"
) -> Dict[str, Any]:
    dev = torch.device(device)
    model.eval()
    model.to(dev)

    # Sample input tensor
    dtype = torch.float16 if next(model.parameters()).dtype == torch.float16 else torch.float32
    input_tensor = torch.randn(batch_size, 3, img_size[0], img_size[1], device=dev, dtype=dtype)

    # Warmup runs
    for _ in range(num_warmup):
        with torch.no_grad():
            if hasattr(model, "tokenizer"):
                _ = model(input_tensor)
            else:
                _ = model(tokenizer, input_tensor)
    if dev.type == "cuda":
        torch.cuda.synchronize()

    # 1. GPU-Only Latency measurement
    latencies_gpu = []
    starter = torch.cuda.Event(enable_timing=True) if dev.type == "cuda" else None
    ender = torch.cuda.Event(enable_timing=True) if dev.type == "cuda" else None

    # Reset max memory tracker
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    for _ in range(num_iterations):
        if dev.type == "cuda":
            starter.record()
            with torch.no_grad():
                if hasattr(model, "tokenizer"):
                    _ = model(input_tensor)
                else:
                    _ = model(tokenizer, input_tensor)
            ender.record()
            torch.cuda.synchronize()
            latencies_gpu.append(starter.elapsed_time(ender)) # milliseconds
        else:
            t0 = time.perf_counter()
            with torch.no_grad():
                if hasattr(model, "tokenizer"):
                    _ = model(input_tensor)
                else:
                    _ = model(tokenizer, input_tensor)
            t1 = time.perf_counter()
            latencies_gpu.append((t1 - t0) * 1000.0)

    # Peak VRAM
    peak_vram_mb = (torch.cuda.max_memory_allocated(dev) / (1024 * 1024)) if dev.type == "cuda" else 0.0

    # 2. Detailed Latency Breakdown (End-to-End, H2D, Encoder, Decoder, Head, D2H)
    cpu_img = torch.randn(batch_size, 3, img_size[0], img_size[1], dtype=dtype)
    breakdown_h2d = []
    breakdown_encoder = []
    breakdown_decoder = []
    breakdown_head = []
    breakdown_d2h = []
    breakdown_e2e = []

    for _ in range(min(num_iterations, 100)):
        t_e2e_start = time.perf_counter()

        # H2D
        t0 = time.perf_counter()
        img_gpu = cpu_img.to(dev, non_blocking=True)
        if dev.type == "cuda": torch.cuda.synchronize()
        t1 = time.perf_counter()
        breakdown_h2d.append((t1 - t0) * 1000.0)

        # Model breakdown
        base = model.model if hasattr(model, "model") else model
        num_steps = base.max_label_length + 1
        pos_queries = base.pos_queries[:, :num_steps].expand(batch_size, -1, -1)
        tgt_in = torch.full((batch_size, 1), tokenizer.bos_id, dtype=torch.long, device=base._device)

        # Encoder
        t0 = time.perf_counter()
        memory = base.encode(img_gpu)
        if dev.type == "cuda": torch.cuda.synchronize()
        t1 = time.perf_counter()
        breakdown_encoder.append((t1 - t0) * 1000.0)

        # Decoder
        t0 = time.perf_counter()
        tgt_out = base.decode(tgt_in, memory, tgt_query=pos_queries)
        if dev.type == "cuda": torch.cuda.synchronize()
        t1 = time.perf_counter()
        breakdown_decoder.append((t1 - t0) * 1000.0)

        # Head
        t0 = time.perf_counter()
        logits = base.head(tgt_out)
        if dev.type == "cuda": torch.cuda.synchronize()
        t1 = time.perf_counter()
        breakdown_head.append((t1 - t0) * 1000.0)

        # D2H
        t0 = time.perf_counter()
        _ = logits.cpu()
        if dev.type == "cuda": torch.cuda.synchronize()
        t1 = time.perf_counter()
        breakdown_d2h.append((t1 - t0) * 1000.0)

        t_e2e_end = time.perf_counter()
        breakdown_e2e.append((t_e2e_end - t_e2e_start) * 1000.0)

    # Compute statistics
    lat_arr = np.array(latencies_gpu)
    mean_ms = float(np.mean(lat_arr))
    median_ms = float(np.median(lat_arr))
    std_ms = float(np.std(lat_arr))
    min_ms = float(np.min(lat_arr))
    max_ms = float(np.max(lat_arr))
    p50_ms = float(np.percentile(lat_arr, 50))
    p90_ms = float(np.percentile(lat_arr, 90))
    p95_ms = float(np.percentile(lat_arr, 95))
    p99_ms = float(np.percentile(lat_arr, 99))
    fps_gpu = float(batch_size * 1000.0 / mean_ms)
    fps_median = float(batch_size * 1000.0 / median_ms)

    e2e_mean_ms = float(np.mean(breakdown_e2e))
    fps_e2e = float(batch_size * 1000.0 / e2e_mean_ms)

    return {
        "batch_size": batch_size,
        "gpu_only": {
            "mean_ms": mean_ms,
            "median_ms": median_ms,
            "std_ms": std_ms,
            "min_ms": min_ms,
            "max_ms": max_ms,
            "p50_ms": p50_ms,
            "p90_ms": p90_ms,
            "p95_ms": p95_ms,
            "p99_ms": p99_ms,
            "fps": fps_gpu,
            "fps_median": fps_median,
            "peak_vram_mb": peak_vram_mb,
        },
        "end_to_end": {
            "mean_ms": e2e_mean_ms,
            "fps": fps_e2e,
            "breakdown_ms": {
                "h2d": float(np.mean(breakdown_h2d)),
                "encoder": float(np.mean(breakdown_encoder)),
                "decoder": float(np.mean(breakdown_decoder)),
                "head": float(np.mean(breakdown_head)),
                "d2h": float(np.mean(breakdown_d2h)),
            }
        }
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="pretrained/parseq_alpr_98.5.ckpt")
    parser.add_argument("--variant", type=str, default="m1")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    system = load_from_checkpoint(args.checkpoint).eval()
    model = create_model_variant(args.variant, system.model)

    res = benchmark_model(
        model=model,
        tokenizer=system.tokenizer,
        batch_size=args.batch_size,
        num_warmup=args.warmup,
        num_iterations=args.iterations,
        device=args.device
    )

    print(f"=== Benchmark Results: {args.variant.upper()} (Batch={args.batch_size}) ===")
    print(f"GPU Latency: mean={res['gpu_only']['mean_ms']:.2f}ms | median={res['gpu_only']['median_ms']:.2f}ms | p95={res['gpu_only']['p95_ms']:.2f}ms")
    print(f"GPU FPS: {res['gpu_only']['fps']:.1f} frames/s (Peak VRAM: {res['gpu_only']['peak_vram_mb']:.1f} MB)")
    print(f"End-to-End Latency: {res['end_to_end']['mean_ms']:.2f}ms (FPS: {res['end_to_end']['fps']:.1f})")
    print(f"Component Breakdown: Encoder={res['end_to_end']['breakdown_ms']['encoder']:.2f}ms | Decoder={res['end_to_end']['breakdown_ms']['decoder']:.2f}ms | Head={res['end_to_end']['breakdown_ms']['head']:.2f}ms")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"Saved to {args.output}")
