import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import time
import json
import argparse
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorrt as trt
from strhub.quant.plugins.trt_plugins import register_parseq_plugins, get_trt_logger
register_parseq_plugins()


TRT_LOGGER = get_trt_logger(trt.Logger.WARNING)


def build_tensorrt_engine(
    onnx_path: str,
    engine_path: str,
    precision: str = "fp32", # "fp32", "fp16", "int8"
    max_batch_size: int = 8,
    img_size: tuple = (32, 128),
    workspace_gb: float = 4.0
) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(os.path.abspath(engine_path)), exist_ok=True)
    t_start = time.perf_counter()

    builder = trt.Builder(TRT_LOGGER)
    if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
    else:
        network = builder.create_network()
    config = builder.create_builder_config()

    # Workspace memory limit (TRT 10/11: set_memory_pool_limit)
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1024 ** 3)))
    else:
        config.max_workspace_size = int(workspace_gb * (1024 ** 3))

    parser = trt.OnnxParser(network, TRT_LOGGER)
    if hasattr(parser, "parse_from_file"):
        success = parser.parse_from_file(onnx_path)
    else:
        with open(onnx_path, "rb") as f:
            success = parser.parse(f.read())
    if not success:
        for error in range(parser.num_errors):
            print("ONNX Parse Error:", parser.get_error(error))
        raise RuntimeError(f"Failed to parse ONNX model: {onnx_path}")

    # Check if network contains custom TRT plugin layers and verify plugin library
    has_custom_plugins = False
    for i in range(network.num_layers):
        l = network.get_layer(i)
        if hasattr(trt, "LayerType") and hasattr(trt.LayerType, "PLUGIN_V2") and l.type == trt.LayerType.PLUGIN_V2:
            has_custom_plugins = True
            break
        elif "plugin" in str(getattr(l, "name", "")).lower() or "plugin" in type(l).__name__.lower():
            has_custom_plugins = True
            break

    if has_custom_plugins:
        from strhub.quant.plugins.trt_plugins import find_plugin_lib_path, get_plugin_dll
        dll = get_plugin_dll(raise_on_error=False)
        if dll is None:
            lib_p = find_plugin_lib_path()
            raise RuntimeError(
                f"TensorRT Engine build requires custom plugin library '{lib_p}', but it could not be loaded.\n"
                f"Please compile the CUDA plugins on this machine using:\n"
                f"  python tools/build_plugins.py\n"
                f"or:\n"
                f"  make plugins"
            )

    # Optimization Profile for Dynamic Batch
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "images",
        min=(1, 3, img_size[0], img_size[1]),
        opt=(1, 3, img_size[0], img_size[1]),
        max=(max_batch_size, 3, img_size[0], img_size[1])
    )
    config.add_optimization_profile(profile)

    # Profiling verbosity: DETAILED preserves layer precision and tactics in inspector
    if hasattr(trt, "ProfilingVerbosity") and hasattr(config, "profiling_verbosity"):
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

    # Precision flags (for TRT versions that support them; in TRT 10/11 Q/DQ nodes control INT8)
    precision = precision.lower().strip()
    if precision in ["fp16", "int8", "int8_io"]:
        if hasattr(trt.BuilderFlag, "FP16"):
            config.set_flag(trt.BuilderFlag.FP16)
    if precision in ["int8", "int8_io"]:
        if hasattr(trt.BuilderFlag, "INT8"):
            # Check if network has explicit Q/DQ nodes or if calibrator is present
            has_qdq = False
            for i in range(network.num_layers):
                l = network.get_layer(i)
                if hasattr(trt, "LayerType") and hasattr(trt.LayerType, "QUANTIZE") and l.type in [trt.LayerType.QUANTIZE, trt.LayerType.DEQUANTIZE]:
                    has_qdq = True
                    break
            has_calibrator = getattr(config, "int8_calibrator", None) is not None
            if has_qdq or has_calibrator:
                config.set_flag(trt.BuilderFlag.INT8)
            else:
                # Provide custom dynamic range so TRT does not fail with calibration error
                for i in range(network.num_inputs):
                    t = network.get_input(i)
                    if hasattr(t, "dynamic_range") and not t.dynamic_range:
                        try:
                            t.dynamic_range = (-128.0, 127.0)
                        except Exception:
                            pass
                for i in range(network.num_layers):
                    l = network.get_layer(i)
                    for j in range(l.num_outputs):
                        t = l.get_output(j)
                        if hasattr(t, "dynamic_range") and not t.dynamic_range:
                            try:
                                t.dynamic_range = (-128.0, 127.0)
                            except Exception:
                                pass
                config.set_flag(trt.BuilderFlag.INT8)

    print(f"Building TensorRT engine ({precision.upper()} | max_batch={max_batch_size}) from {onnx_path}...")
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(f"Failed to build TensorRT engine for {onnx_path}")

    with open(engine_path, "wb") as f:
        f.write(plan)

    t_end = time.perf_counter()
    build_time_s = t_end - t_start
    engine_size_mb = os.path.getsize(engine_path) / (1024 * 1024)

    # Clean up builder resources
    del plan, network, parser, config, profile, builder
    import gc
    gc.collect()

    print(f"Engine built successfully in {build_time_s:.2f}s: {engine_path} ({engine_size_mb:.2f} MB)")

    return {
        "engine_path": engine_path,
        "onnx_path": onnx_path,
        "precision": precision,
        "max_batch_size": max_batch_size,
        "build_time_seconds": build_time_s,
        "engine_size_mb": engine_size_mb,
        "workspace_gb": workspace_gb,
    }


def build_all_engines(max_batch_size: int = 64):
    """Builds TensorRT engines for all available exported ONNX models."""
    targets = [
        ("onnx/parseq_m0_ar_fp32.onnx", "trt/parseq_m0_ar_fp32.engine", "fp32"),
        ("onnx/parseq_m1_nar_fp32.onnx", "trt/parseq_m1_nar_fp32.engine", "fp32"),
        ("onnx/parseq_m2_nar_fp16.onnx", "trt/parseq_m2_nar_fp16.engine", "fp16"),
        ("onnx/parseq_m3_nar_int8_naive.onnx", "trt/parseq_m3_nar_int8_naive.engine", "int8"),
        ("onnx/parseq_m4_nar_int8_ptq.onnx", "trt/parseq_m4_nar_int8_ptq.engine", "int8"),
        ("onnx/parseq_m5_nar_int8_io_ptq.onnx", "trt/parseq_m5_nar_int8_io_ptq.engine", "int8"),
        ("onnx/parseq_m6_nar_int8_io_qat.onnx", "trt/parseq_m6_nar_int8_io_qat.engine", "int8"),
    ]
    results = {}
    for onnx_p, eng_p, prec in targets:
        if not os.path.exists(onnx_p):
            print(f"Skipping {onnx_p} (not found). Run export_onnx.py first.")
            continue
        print(f"\n==================================================")
        print(f"Building Engine: {onnx_p} -> {eng_p} ({prec.upper()} | max_batch={max_batch_size})")
        print(f"==================================================")
        try:
            res = build_tensorrt_engine(onnx_path=onnx_p, engine_path=eng_p, precision=prec, max_batch_size=max_batch_size)
            results[eng_p] = res
        except Exception as e:
            print(f"Error building {eng_p}: {e}")
            results[eng_p] = f"FAILED: {e}"
    print("\n=== TensorRT Build Summary ===")
    for k, v in results.items():
        if isinstance(v, dict):
            print(f"  {os.path.basename(k)}: {v['engine_size_mb']:.2f} MB in {v['build_time_seconds']:.1f}s")
        else:
            print(f"  {os.path.basename(k)}: {v}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16", "int8", "int8_io"])
    parser.add_argument("--max_batch", type=int, default=64)
    parser.add_argument("--workspace_gb", type=float, default=4.0, help="Workspace memory limit in GB")
    parser.add_argument("--all", action="store_true", help="Build all engines for m0..m6")
    args = parser.parse_args()

    if args.all:
        build_all_engines(max_batch_size=args.max_batch)
    else:
        if not args.onnx or not args.output:
            parser.error("--onnx and --output are required unless --all is specified.")
        build_tensorrt_engine(
            onnx_path=args.onnx,
            engine_path=args.output,
            precision=args.precision,
            max_batch_size=args.max_batch,
            workspace_gb=args.workspace_gb,
        )
