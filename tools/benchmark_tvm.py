#!/usr/bin/env python3
"""Statistical Performance Benchmarker for Apache TVM Compiled Libraries.
Measures inference latency, throughput (FPS), percentiles (p50..p99), and memory usage
with synchronized warmup on GPU Tensor Cores / CUDA or CPU AVX2.
Mirrors architectural parity with tools/benchmark_tensorrt.py.
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
import json
import argparse
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import tvm
from strhub.models.tvm_utils import TVMRuntimeSession, get_tvm_device


def synchronize_device(device_str: str):
    """Synchronizes device queue ensuring accurate timing measurements."""
    if "cuda" in device_str.lower() and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif "cuda" in device_str.lower() and tvm.cuda().exist:
        tvm.cuda().sync()


def benchmark_tvm(
    lib_path: str,
    batch_size: int = 1,
    img_size: tuple = (32, 128),
    num_warmup: int = 50,
    num_iterations: int = 200,
    device: str = "cuda"
) -> Dict[str, Any]:
    """Runs statistical benchmark on an Apache TVM compiled library."""
    if not os.path.isfile(lib_path):
        raise FileNotFoundError(f"Biblioteca TVM não encontrada em: {lib_path}")

    # Use CPU if CUDA not available on hardware
    if "cuda" in device.lower() and not torch.cuda.is_available() and not tvm.cuda().exist:
        print("[Aviso] GPU CUDA não disponível. Executando benchmark TVM em CPU.")
        device = "cpu"

    session = TVMRuntimeSession(lib_path, device=device)
    input_shape = (batch_size, 3, img_size[0], img_size[1])
    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    # Reset CUDA memory stats if available
    peak_vram_mb = 0.0
    if "cuda" in device.lower() and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # 1. Warm-up sincronizado
    print(f"Executando warmup sincronizado ({num_warmup} iterações)...")
    for _ in range(num_warmup):
        try:
            _ = session.run(dummy_input)
        except Exception:
            # Em caso de batch 1 estático, executa amostra a amostra
            for b in range(batch_size):
                _ = session.run(dummy_input[b:b+1])
        synchronize_device(device)

    # 2. Medição estatística de latência
    print(f"Executando profiling estatístico ({num_iterations} iterações, batch={batch_size})...")
    latencies = []

    use_cuda_events = "cuda" in device.lower() and torch.cuda.is_available()
    if use_cuda_events:
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        for _ in range(num_iterations):
            starter.record()
            try:
                _ = session.run(dummy_input)
            except Exception:
                for b in range(batch_size):
                    _ = session.run(dummy_input[b:b+1])
            ender.record()
            torch.cuda.synchronize()
            latencies.append(starter.elapsed_time(ender))
    else:
        for _ in range(num_iterations):
            t0 = time.perf_counter()
            try:
                _ = session.run(dummy_input)
            except Exception:
                for b in range(batch_size):
                    _ = session.run(dummy_input[b:b+1])
            synchronize_device(device)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    # Coleta de memória VRAM
    if "cuda" in device.lower() and torch.cuda.is_available():
        peak_vram_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
    else:
        # RSS em CPU
        try:
            import resource
            peak_vram_mb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)
        except Exception:
            peak_vram_mb = 0.0

    lat_arr = np.array(latencies)
    mean_ms = float(np.mean(lat_arr))
    median_ms = float(np.median(lat_arr))
    std_ms = float(np.std(lat_arr))
    min_ms = float(np.min(lat_arr))
    max_ms = float(np.max(lat_arr))
    p50_ms = float(np.percentile(lat_arr, 50))
    p90_ms = float(np.percentile(lat_arr, 90))
    p95_ms = float(np.percentile(lat_arr, 95))
    p99_ms = float(np.percentile(lat_arr, 99))
    fps = float(batch_size * 1000.0 / mean_ms)
    fps_median = float(batch_size * 1000.0 / median_ms)

    # Metadados da biblioteca
    manifest_file = lib_path + ".manifest.json"
    manifest_data = {}
    if os.path.exists(manifest_file):
        try:
            with open(manifest_file, "r", encoding="utf-8") as mf:
                manifest_data = json.load(mf)
        except Exception:
            pass

    res = {
        "lib_path": lib_path,
        "device": device,
        "batch_size": batch_size,
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "std_ms": std_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "p50_ms": p50_ms,
        "p90_ms": p90_ms,
        "p95_ms": p95_ms,
        "p99_ms": p99_ms,
        "fps": fps,
        "fps_median": fps_median,
        "peak_vram_mb": peak_vram_mb,
        "target": manifest_data.get("target", "N/A"),
        "precision": manifest_data.get("precision", "N/A"),
        "backend_mode": manifest_data.get("backend_mode", "N/A"),
    }

    # Limpeza explícita de recursos
    del session, dummy_input
    if "cuda" in device.lower() and torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()

    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apache TVM Performance Benchmarker")
    parser.add_argument("--lib", type=str, required=True, help="Caminho da biblioteca TVM (.so / .tar)")
    parser.add_argument("--batch_size", type=int, default=1, help="Dimensão do batch")
    parser.add_argument("--warmup", type=int, default=50, help="Iterações de warm-up")
    parser.add_argument("--iterations", type=int, default=200, help="Iterações cronometradas")
    parser.add_argument("--device", type=str, default="cuda", help="Dispositivo de execução (cuda / cpu)")
    parser.add_argument("--output", type=str, default=None, help="Caminho do arquivo JSON de saída")
    args = parser.parse_args()

    res = benchmark_tvm(
        lib_path=args.lib,
        batch_size=args.batch_size,
        num_warmup=args.warmup,
        num_iterations=args.iterations,
        device=args.device,
    )

    print(f"\n=== Apache TVM Benchmark ({os.path.basename(args.lib)} | Batch={res['batch_size']}) ===")
    print(f"Dispositivo: {res['device']} | Backend: {res['backend_mode']} | Precisão: {res['precision'].upper()}")
    print(f"Latência: média={res['mean_ms']:.2f}ms | mediana={res['median_ms']:.2f}ms | p95={res['p95_ms']:.2f}ms | p99={res['p99_ms']:.2f}ms")
    print(f"Throughput: {res['fps']:.1f} FPS (mediana: {res['fps_median']:.1f} FPS)")
    print(f"Memória Pico: {res['peak_vram_mb']:.1f} MB")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"Resultados salvos em: {args.output}")
