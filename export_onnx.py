#!/usr/bin/env python3
"""Export and Quantize PARSeq to ONNX (FP32) and ONNX INT8."""

import argparse
from pathlib import Path
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
    args = parser.parse_args()

    Path(args.output_fp32).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_int8).parent.mkdir(parents=True, exist_ok=True)

    precision_str = "FP16" if args.fp16 else "FP32"
    print(f"[1/2] Exportando PARSeq ({args.mode.upper()}) para ONNX {precision_str}: {args.output_fp32} (device={args.device})...")
    PARSeqQuantizer.export_onnx(
        args.checkpoint,
        args.output_fp32,
        mode=args.mode,
        device=args.device,
        fp16=args.fp16,
        max_label_length=args.max_label_length,
    )
    print(f"✓ ONNX FP32 salvo com sucesso em: {args.output_fp32}")

    print(f"[2/2] Quantizando grafo para INT8 físico: {args.output_int8}...")
    PARSeqQuantizer.export_onnx_int8(
        args.output_fp32,
        args.output_int8,
        op_types_to_quantize=["MatMul"],
    )
    print(f"✓ ONNX INT8 salvo com sucesso em: {args.output_int8}")
    print("\nModelos ONNX FP32 e INT8 gerados com sucesso em outputs/!")


if __name__ == "__main__":
    main()
