import os
import sys
import time
import json
import argparse
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import torch
import tensorrt as trt
from strhub.quant.plugins.trt_plugins import register_parseq_plugins
register_parseq_plugins()


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def benchmark_tensorrt(
    engine_path: str,
    batch_size: int = 1,
    img_size: tuple = (32, 128),
    num_warmup: int = 50,
    num_iterations: int = 200
) -> Dict[str, Any]:
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    # Configure batch shape
    input_shape = (batch_size, 3, img_size[0], img_size[1])
    if not context.set_input_shape("images", input_shape):
        min_s, opt_s, max_s = engine.get_tensor_profile_shape("images", 0)
        raise RuntimeError(
            f"Não é possível configurar batch_size={batch_size} no engine '{os.path.basename(engine_path)}'. "
            f"O perfil de otimização compilado aceita intervalo de batch: min={min_s[0]}, opt={opt_s[0]}, max={max_s[0]}."
        )

    output_shape = tuple(context.get_tensor_shape("logits"))

    # Inspect engine info
    inspector = engine.create_engine_inspector()
    info_json = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    layer_types = {}
    try:
        engine_info = json.loads(info_json)
        layers = engine_info.get("Layers", [])
        total_layers = len(layers)
        for l in layers:
            t = l.get("LayerType", l.get("type", "Unknown"))
            layer_types[t] = layer_types.get(t, 0) + 1
    except Exception:
        total_layers = -1

    # Allocate CUDA memory matching engine data type
    in_dtype = engine.get_tensor_dtype("images")
    torch_in_dtype = torch.float16 if in_dtype == trt.DataType.HALF else torch.float32
    d_input = torch.randn(input_shape, dtype=torch_in_dtype, device="cuda")

    out_dtype = engine.get_tensor_dtype("logits")
    torch_out_dtype = torch.float16 if out_dtype == trt.DataType.HALF else torch.float32
    d_output = torch.empty(output_shape, dtype=torch_out_dtype, device="cuda")

    context.set_tensor_address("images", int(d_input.data_ptr()))
    context.set_tensor_address("logits", int(d_output.data_ptr()))

    cuda_stream = torch.cuda.Stream()
    stream = cuda_stream.cuda_stream

    # Warmup
    for _ in range(num_warmup):
        context.execute_async_v3(stream)
    cuda_stream.synchronize()

    # Latency measurement
    latencies = []
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    for _ in range(num_iterations):
        starter.record(cuda_stream)
        context.execute_async_v3(stream)
        ender.record(cuda_stream)
        cuda_stream.synchronize()
        latencies.append(starter.elapsed_time(ender))

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

    return {
        "engine_path": engine_path,
        "batch_size": batch_size,
        "total_layers": total_layers,
        "layer_types": layer_types,
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
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    res = benchmark_tensorrt(
        engine_path=args.engine,
        batch_size=args.batch_size,
        num_warmup=args.warmup,
        num_iterations=args.iterations
    )

    print(f"=== TensorRT Benchmark ({os.path.basename(args.engine)} | Batch={res['batch_size']}) ===")
    print(f"Latency: mean={res['mean_ms']:.2f}ms | median={res['median_ms']:.2f}ms | p95={res['p95_ms']:.2f}ms | p99={res['p99_ms']:.2f}ms")
    print(f"Throughput: {res['fps']:.1f} FPS (median-based: {res['fps_median']:.1f} FPS)")
    print(f"Compiled Layers: {res.get('total_layers', 'N/A')} {res.get('layer_types', {})}")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
