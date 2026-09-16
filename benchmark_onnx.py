#!/usr/bin/env python3
"""Benchmark latency, FPS, and percentiles for an ONNX model."""

import argparse
import os
import time
import numpy as np
import onnxruntime as ort


def main():
    parser = argparse.ArgumentParser(description="Benchmark ONNX Model Performance")
    parser.add_argument("model_path", default="outputs/parseq_nar_int8.onnx", nargs="?", help="Path to .onnx file")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Execution device")
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
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if args.device == "cuda" else ["CPUExecutionProvider"]
    
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
