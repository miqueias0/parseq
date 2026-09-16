#!/usr/bin/env python3
"""Benchmark latency, FPS, and percentiles for an ONNX model."""

import argparse
import os
import sys
import time
import numpy as np


def setup_tensorrt_library_paths():
    """Auto-detects and loads TensorRT, cuDNN, and cuBLAS shared libraries from pip packages."""
    import ctypes

    lib_dirs = []
    # Check well-known virtualenv path first
    for candidate in [
        "/home/mon25/modelos/.venv/lib/python3.12/site-packages",
        os.path.join(sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages"),
    ]:
        if os.path.isdir(candidate) and candidate not in sys.path:
            sys.path.append(candidate)

    for base in list(sys.path):
        if not os.path.isdir(base):
            continue
        trt_dir = os.path.join(base, "tensorrt_libs")
        if os.path.isdir(trt_dir) and trt_dir not in lib_dirs:
            lib_dirs.append(trt_dir)
        nvidia_dir = os.path.join(base, "nvidia")
        if os.path.isdir(nvidia_dir):
            for sub in ["cuda_runtime", "cublas", "cudnn", "curand", "cufft", "cusolver", "cusparse"]:
                sub_lib = os.path.join(nvidia_dir, sub, "lib")
                if os.path.isdir(sub_lib) and sub_lib not in lib_dirs:
                    lib_dirs.append(sub_lib)

    if lib_dirs:
        old_ld = os.environ.get("LD_LIBRARY_PATH", "")
        new_ld = ":".join(lib_dirs) + (f":{old_ld}" if old_ld else "")
        os.environ["LD_LIBRARY_PATH"] = new_ld

        libs_to_load = [
            "libcudart.so.12",
            "libcublasLt.so.12",
            "libcublas.so.12",
            "libcudnn.so.9",
            "libcurand.so.10",
            "libcufft.so.11",
            "libnvinfer.so.10",
            "libnvinfer_plugin.so.10",
            "libnvonnxparser.so.10",
        ]
        for lib_name in libs_to_load:
            for lib_dir in lib_dirs:
                lib_path = os.path.join(lib_dir, lib_name)
                if os.path.exists(lib_path):
                    try:
                        ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
                        break
                    except Exception:
                        pass


setup_tensorrt_library_paths()
import onnxruntime as ort


def main():
    parser = argparse.ArgumentParser(description="Benchmark ONNX Model Performance")
    parser.add_argument("model_path", default="outputs/parseq_nar_int8.onnx", nargs="?", help="Path to .onnx file")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Execution device")
    parser.add_argument(
        "--provider",
        default="auto",
        choices=["auto", "tensorrt", "cuda", "cpu"],
        help="ONNX Runtime Execution Provider (auto, tensorrt, cuda, cpu)",
    )
    parser.add_argument("--iterations", type=int, default=100, help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=15, help="Number of warmup iterations")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--height", type=int, default=32, help="Image height")
    parser.add_argument("--width", type=int, default=128, help="Image width")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"ONNX model file not found: {args.model_path}")

    file_size_mb = os.path.getsize(args.model_path) / (1024.0 * 1024.0)

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 4

    # Build Execution Providers list
    trt_cache_dir = os.path.join(os.path.dirname(args.model_path) or ".", "trt_cache")
    os.makedirs(trt_cache_dir, exist_ok=True)
    trt_options = {
        "trt_fp16_enable": True,
        "trt_int8_enable": True,
        "trt_max_workspace_size": 2147483648,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": trt_cache_dir,
    }

    available = ort.get_available_providers()
    if args.provider == "tensorrt" or (args.provider == "auto" and "TensorrtExecutionProvider" in available and args.device == "cuda"):
        providers = [("TensorrtExecutionProvider", trt_options), "CUDAExecutionProvider", "CPUExecutionProvider"]
    elif args.device == "cuda" and (args.provider in ["cuda", "auto"]):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    print(f"Configuring ONNX session with providers: {[p[0] if isinstance(p, tuple) else p for p in providers]}")
    session = ort.InferenceSession(args.model_path, sess_opts, providers=providers)
    active_provider = session.get_providers()[0]
    input_name = session.get_inputs()[0].name

    dummy_input = np.random.randn(args.batch_size, 3, args.height, args.width).astype(np.float32)

    # Warm-up
    for _ in range(args.warmup):
        _ = session.run(None, {input_name: dummy_input})

    # Timing loop
    latencies_ms = []
    for _ in range(args.iterations):
        t0 = time.perf_counter()
        _ = session.run(None, {input_name: dummy_input})
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    lat = np.array(latencies_ms)
    fps = (args.batch_size * 1000.0) / lat.mean()

    print(f"\n{'='*60}")
    print(f"  ONNX Performance Benchmark: {os.path.basename(args.model_path)}")
    print(f"{'='*60}")
    print(f"  Execution Provider:  {active_provider}")
    print(f"  Model File Size:     {file_size_mb:.2f} MB")
    print(f"  Batch Size:          {args.batch_size}")
    print(f"  Input Resolution:    3x{args.height}x{args.width}")
    print(f"  Iterations:          {args.iterations} (Warmup: {args.warmup})")
    print(f"{'-'*60}")
    print(f"  Latency (Mean):      {lat.mean():.2f} ms")
    print(f"  Latency (Median P50):{np.percentile(lat, 50):.2f} ms")
    print(f"  Latency (P90):       {np.percentile(lat, 90):.2f} ms")
    print(f"  Latency (P99):       {np.percentile(lat, 99):.2f} ms")
    print(f"  Throughput:          {fps:.2f} FPS")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
