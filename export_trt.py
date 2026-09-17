#!/usr/bin/env python3
"""
export_trt.py - Export PARSeq to ONNX (Q/DQ) and TensorRT INT8 Engine
====================================================================
1. Exports PARSeq encoder with strict Q/DQ placement for TensorRT fusion:
   - QKV GEMM fusion
   - SkipLayerNorm fusion
   - Fast GELU fusion
2. Compiles to high-performance TensorRT INT8 Engine (.engine / .plan)
"""

import argparse
import os
import torch

from strhub.models.utils import create_model
from strhub.quantization.trt_exporter import TensorRTExporter


def parse_args():
    parser = argparse.ArgumentParser(description="Export PARSeq to ONNX and TensorRT Engine")
    parser.add_argument("--checkpoint", type=str, default=None, help="Pretrained model checkpoint")
    parser.add_argument("--onnx_path", type=str, default="parseq_qdq.onnx", help="Path to output ONNX file")
    parser.add_argument("--engine_path", type=str, default="parseq_int8.engine", help="Path to output TensorRT engine")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--int8", action="store_true", default=True, help="Enable TensorRT INT8 mode")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (>=13)")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 65)
    print("PARSeq TensorRT INT8 Pipeline Exporter")
    print("=" * 65)
    print(f"Device: {args.device} | Target ONNX: {args.onnx_path} | Target Engine: {args.engine_path}")

    # Load model
    model = create_model("parseq", pretrained=False)
    if args.checkpoint and os.path.isfile(args.checkpoint):
        print(f"[*] Loading checkpoint from {args.checkpoint}...")
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        clean_state = {k.replace("model.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(clean_state, strict=False)

    exporter = TensorRTExporter(model)

    # 1. Export ONNX graph with Q/DQ pairing
    onnx_file = exporter.export_onnx(
        output_path=args.onnx_path,
        input_shape=(1, 3, 32, 128),
        opset_version=args.opset,
        device=args.device,
    )

    # 2. Build TensorRT engine if TensorRT is installed and GPU is available
    if torch.cuda.is_available():
        engine_file = exporter.build_trt_engine(
            onnx_path=onnx_file,
            engine_path=args.engine_path,
            min_shape=(1, 3, 32, 128),
            opt_shape=(8, 3, 32, 128),
            max_shape=(32, 3, 32, 128),
            int8_mode=args.int8,
        )
        if engine_file:
            print(f"[+] Engine compilation succeeded: {engine_file}")
        else:
            print("[!] TensorRT engine compilation skipped or failed.")
    else:
        print("[*] CUDA is not available on this machine. Generated ONNX model is ready for TensorRT compilation on target GPU.")

    print("=" * 65)


if __name__ == "__main__":
    main()
