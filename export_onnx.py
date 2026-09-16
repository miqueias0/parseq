#!/usr/bin/env python3
"""Export and Quantize PARSeq to ONNX (FP32) and ONNX INT8."""

import argparse
import os
import sys
from pathlib import Path


def setup_tensorrt_library_paths():
    """Auto-detects and loads TensorRT, cuDNN, and cuBLAS shared libraries from pip packages."""
    import ctypes

    lib_dirs = []
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
from strhub.models.quantization import PARSeqQuantizer


def main():
    parser = argparse.ArgumentParser(description="Export PARSeq to ONNX (FP32) and ONNX INT8")
    parser.add_argument("checkpoint", default="pretrained=parseq", nargs="?", help="Model checkpoint (default: pretrained=parseq)")
    parser.add_argument("--output_fp32", default="outputs/parseq_nar.onnx", help="Output path for FP32 ONNX model")
    parser.add_argument("--output_int8", default="outputs/parseq_nar_int8.onnx", help="Output path for INT8 ONNX model")
    parser.add_argument("--mode", default="nar", choices=["nar", "ar"], help="Decoding mode: 'nar' (fastest) or 'ar'")
    parser.add_argument("--device", default="cuda", help="Device to trace model on (cuda or cpu)")
    parser.add_argument("--max_label_length", type=int, default=25, help="Maximum label length")
    parser.add_argument("--fp16", action="store_true", default=False, help="Export in FP16 for native NVIDIA Tensor Cores")
    parser.add_argument("--qdq", action="store_true", default=False, help="Export in Static QDQ INT8 format for NVIDIA TensorRT")
    parser.add_argument("--output_qdq", default="outputs/parseq_nar_qdq.onnx", help="Output path for Static QDQ ONNX model")
    parser.add_argument("--calib_dir", default=None, help="Directory of calibration images (e.g. data/test or demo_images)")
    parser.add_argument("--calib_samples", type=int, default=64, help="Number of calibration samples")
    parser.add_argument("--calib_method", default="MinMax", choices=["MinMax", "Entropy", "Percentile"], help="Calibration method")
    parser.add_argument(
        "--quant_method",
        default="none",
        choices=["none", "ibert", "unified_int8", "real_int8", "int_flashattn", "jetfire_fqt"],
        help="Quantization method to apply before ONNX export (ibert, unified_int8, real_int8, etc.)",
    )
    parser.add_argument("--block_size", type=int, default=32, help="Block size for Jetfire / INT-FlashAttention")
    args = parser.parse_args()

    Path(args.output_fp32).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_int8).parent.mkdir(parents=True, exist_ok=True)

    if args.quant_method != "none":
        from strhub.models.utils import load_from_checkpoint
        print(f"Loading checkpoint '{args.checkpoint}' and applying {args.quant_method.upper()}...")
        kwargs = {}
        if not args.checkpoint.startswith("pretrained="):
            kwargs["max_label_length"] = args.max_label_length
        target_model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)
        target_model = PARSeqQuantizer.quantize(target_model, method=args.quant_method, block_size=args.block_size, inplace=True)
        if args.quant_method != "dynamic":
            target_model = target_model.to(args.device)
    else:
        target_model = args.checkpoint

    precision_str = "FP16" if args.fp16 else ("INT8 (" + args.quant_method.upper() + ")" if args.quant_method != "none" else "FP32")
    print(f"[1/2] Exportando PARSeq ({args.mode.upper()}) para ONNX {precision_str}: {args.output_fp32} (device={args.device})...")
    export_kwargs = {
        "mode": args.mode,
        "device": args.device,
        "fp16": args.fp16,
    }
    if not (isinstance(target_model, str) and target_model.startswith("pretrained=")):
        export_kwargs["max_label_length"] = args.max_label_length

    PARSeqQuantizer.export_onnx(
        target_model,
        args.output_fp32,
        **export_kwargs,
    )
    print(f"✓ ONNX {precision_str} salvo com sucesso em: {args.output_fp32}")

    if args.qdq:
        Path(args.output_qdq).parent.mkdir(parents=True, exist_ok=True)
        print(f"[2/2] Quantizando para Static QDQ INT8 (TensorRT format): {args.output_qdq}...")
        PARSeqQuantizer.export_onnx_int8_qdq(
            float_onnx_path=args.output_fp32,
            output_qdq_path=args.output_qdq,
            calib_dir=args.calib_dir,
            calib_samples=args.calib_samples,
            calibrate_method=args.calib_method,
        )
        print(f"✓ ONNX Static QDQ INT8 salvo com sucesso em: {args.output_qdq}")
    else:
        print(f"[2/2] Quantizando grafo para Dynamic INT8: {args.output_int8}...")
        PARSeqQuantizer.export_onnx_int8(
            args.output_fp32,
            args.output_int8,
            op_types_to_quantize=["MatMul"],
        )
        print(f"✓ ONNX INT8 salvo com sucesso em: {args.output_int8}")

    print("\nModelos ONNX gerados com sucesso em outputs/!")


if __name__ == "__main__":
    main()
