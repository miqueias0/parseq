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
    parser.add_argument("--qdq", action="store_true", default=False, help="Export in Static QDQ INT8 format for NVIDIA TensorRT")
    parser.add_argument("--output_qdq", default="outputs/parseq_nar_qdq.onnx", help="Output path for Static QDQ ONNX model")
    parser.add_argument("--calib_dir", default=None, help="Directory of calibration images (e.g. data/test or demo_images)")
    parser.add_argument("--calib_samples", type=int, default=64, help="Number of calibration samples")
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
        target_model = load_from_checkpoint(args.checkpoint, max_label_length=args.max_label_length).eval().to(args.device)
        target_model = PARSeqQuantizer.quantize(target_model, method=args.quant_method, block_size=args.block_size, inplace=True)
        if args.quant_method != "dynamic":
            target_model = target_model.to(args.device)
    else:
        target_model = args.checkpoint

    precision_str = "FP16" if args.fp16 else ("INT8 (" + args.quant_method.upper() + ")" if args.quant_method != "none" else "FP32")
    print(f"[1/2] Exportando PARSeq ({args.mode.upper()}) para ONNX {precision_str}: {args.output_fp32} (device={args.device})...")
    PARSeqQuantizer.export_onnx(
        target_model,
        args.output_fp32,
        mode=args.mode,
        device=args.device,
        fp16=args.fp16,
        max_label_length=args.max_label_length,
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
