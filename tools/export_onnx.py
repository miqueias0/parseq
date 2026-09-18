import os
import sys

# Ensure UTF-8 output encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import copy
import json
import argparse
import hashlib
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import onnx

from strhub.models.utils import load_from_checkpoint
from strhub.models.parseq.quantized_parseq import create_model_variant


class ONNXExportWrapper(nn.Module):
    """Wrapper that exposes clean forward(images) -> logits for ONNX export."""
    def __init__(self, model: nn.Module, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        base = self.model.model if hasattr(self.model, "model") else self.model
        num_steps = base.max_label_length + 1
        memory = base.encode(images)
        # Broadcast pos_queries dynamically across batch dimension
        batch_dummy = torch.zeros_like(images[:, :1, 0, 0]).unsqueeze(-1)
        pos_queries = base.pos_queries[:, :num_steps] + batch_dummy

        if getattr(base, "decode_ar", False):
            tgt_mask = torch.triu(torch.ones((num_steps, num_steps), dtype=torch.bool, device=images.device), 1)
            query_mask = tgt_mask
            tokens = [torch.full_like(images[:, :1, 0, 0].long(), self.tokenizer.bos_id)]
            logits = []
            for i in range(num_steps):
                j = i + 1
                curr_tgt = torch.cat(tokens, dim=1)
                tgt_out = base.decode(
                    curr_tgt,
                    memory,
                    tgt_mask[:j, :j],
                    tgt_query=pos_queries[:, i:j],
                    tgt_query_mask=query_mask[i:j, :j],
                )
                p_i = base.head(tgt_out)
                logits.append(p_i)
                if j < num_steps:
                    next_tok = p_i.argmax(-1)
                    tokens.append(next_tok)
            return torch.cat(logits, dim=1)
        else:
            tgt_in = torch.full_like(images[:, :1, 0, 0].long(), self.tokenizer.bos_id)
            tgt_out = base.decode(tgt_in, memory, tgt_query=pos_queries)
            logits = base.head(tgt_out)
            return logits


def generate_model_fingerprint(model: nn.Module) -> Dict[str, Any]:
    """Generates an architecture fingerprint/manifest before export (Section 41)."""
    base = model.model if hasattr(model, "model") else model
    param_count = sum(p.numel() for p in model.parameters())
    manifest = {
        "architecture": "PARSeq-AR" if getattr(base, "decode_ar", False) else "PARSeq-NAR",
        "embed_dim": base.encoder.embed_dim,
        "encoder_depth": len(base.encoder.blocks),
        "decoder_depth": len(base.decoder.layers),
        "max_label_length": base.max_label_length,
        "decode_ar": getattr(base, "decode_ar", False),
        "refine_iters": getattr(base, "refine_iters", 0),
        "total_parameters": param_count,
    }
    manifest_str = json.dumps(manifest, sort_keys=True)
    manifest["fingerprint_sha256"] = hashlib.sha256(manifest_str.encode()).hexdigest()
    return manifest


def export_onnx(
    checkpoint_path: str = "pretrained/parseq_alpr_98.5.ckpt",
    variant: str = "m1",
    output_path: str = "onnx/parseq_nar.onnx",
    batch_size: int = 1,
    img_size: tuple = (32, 128),
    opset_version: int = 18,
    dynamic_batch: bool = True,
    fuse_shapes: bool = False,
    fuse_mha: bool = False,
    fuse_mlp: bool = False,
    fuse_layernorm: bool = False,
    use_int_flashattention: bool = False,
    use_plugin: bool = False,
) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    device = torch.device("cpu") # Export from CPU for broad ONNX converter compatibility

    system = load_from_checkpoint(checkpoint_path).eval().to(device)
    variant = variant.lower().strip()

    from strhub.models.parseq.quantized_parseq import QuantizedLinear
    # Enable explicit Q/DQ export for quantized variants
    QuantizedLinear.global_export_qdq = (variant in ["m4", "m5", "m6"])

    calib_file = "results/calibration/calibration_stats.json"
    is_already_quant = any(isinstance(m, QuantizedLinear) for m in system.modules())
    has_custom_fusion = (fuse_mha or fuse_mlp or fuse_layernorm or use_int_flashattention or use_plugin)
    if is_already_quant and variant == "m6" and not has_custom_fusion:
        model = copy.deepcopy(system.model).eval().to(device)
    else:
        model = create_model_variant(
            variant,
            system.model,
            calibration_file=calib_file if os.path.exists(calib_file) else None,
            fuse_mha=fuse_mha,
            fuse_mlp=fuse_mlp,
            fuse_layernorm=fuse_layernorm,
            use_int_flashattention=use_int_flashattention,
            use_plugin=use_plugin,
        ).eval().to(device)

    # Load fine-tuned weights for M6
    if variant == "m6":
        qat_ckpt_path = checkpoint_path if ("qat" in checkpoint_path.lower() or "m6" in checkpoint_path.lower()) else "pretrained/parseq_alpr_qat_m6.ckpt"
        if os.path.exists(qat_ckpt_path):
            ckpt = torch.load(qat_ckpt_path, map_location=device, weights_only=False)
            sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
            clean_sd = {k.replace("model.", ""): v for k, v in sd.items()}
            model.load_state_dict(clean_sd, strict=False)
            for m in model.modules():
                if isinstance(m, QuantizedLinear):
                    m.recompute_weight_scale()
            if os.path.exists(calib_file):
                from strhub.models.parseq.quantized_parseq import load_calibration_into_model
                load_calibration_into_model(model, calib_file)
            print(f"Loaded trained QAT checkpoint from {qat_ckpt_path} for variant M6.")
    elif variant == "m2":
        model = model.float()

    wrapper = ONNXExportWrapper(model, system.tokenizer).eval().to(device)

    # Generate and save fingerprint manifest
    manifest = generate_model_fingerprint(model)
    manifest_path = output_path + ".manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    dummy_input = torch.randn(batch_size, 3, img_size[0], img_size[1], dtype=torch.float32, device=device)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "images": {0: "batch_size"},
            "logits": {0: "batch_size"},
        }

    print(f"Exporting variant {variant.upper()} to ONNX (opset {opset_version}, Q/DQ={QuantizedLinear.global_export_qdq})...")
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        dynamo=False
    )

    # Reset global_export_qdq flag
    QuantizedLinear.global_export_qdq = False

    # Check model validity
    if not use_plugin:
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)

    if fuse_shapes and not use_plugin:
        try:
            import onnxsim
            print(f"Simplifying ONNX graph with onnxsim (--fuse_shapes active)...")
            simplified_model, check = onnxsim.simplify(
                output_path,
                test_input_shapes={"images": [1, 3, img_size[0], img_size[1]]}
            )
            if check:
                onnx.save(simplified_model, output_path)
                print(f"Successfully simplified graph with onnxsim.")
            else:
                print("Warning: onnxsim check failed, keeping original model.")
        except Exception as e:
            print(f"Warning: could not run onnxsim simplify: {e}")

    # If M2, convert to true FP16 format
    if variant == "m2":
        try:
            from onnxconverter_common import float16
            m = onnx.load(output_path)
            m_fp16 = float16.convert_float_to_float16(m, keep_io_types=False)
            # Ensure Cast nodes target FLOAT16 (10) instead of FLOAT (1)
            for n in m_fp16.graph.node:
                if n.op_type == "Cast":
                    for attr in n.attribute:
                        if attr.name == "to" and attr.i == 1:
                            attr.i = 10
            onnx.save(m_fp16, output_path)
            print(f"Converted {output_path} to FP16 with onnxconverter_common.")
        except Exception as e:
            print(f"Warning: could not convert {output_path} to FP16: {e}")

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Exported successfully to {output_path} ({file_size_mb:.2f} MB).")
    return output_path


def export_all_variants(
    checkpoint_path: str = "pretrained/parseq_alpr_98.5.ckpt",
    opset: int = 18,
    fuse_shapes: bool = False,
    fuse_mha: bool = False,
    fuse_mlp: bool = False,
    fuse_layernorm: bool = False,
):
    """Exports all variants m0 through m6 to onnx/ directory."""
    variants = [
        ("m0", "onnx/parseq_m0_ar_fp32.onnx"),
        ("m1", "onnx/parseq_m1_nar_fp32.onnx"),
        ("m2", "onnx/parseq_m2_nar_fp16.onnx"),
        ("m3", "onnx/parseq_m3_nar_int8_naive.onnx"),
        ("m4", "onnx/parseq_m4_nar_int8_ptq.onnx"),
        ("m5", "onnx/parseq_m5_nar_int8_io_ptq.onnx"),
        ("m6", "onnx/parseq_m6_nar_int8_io_qat.onnx"),
    ]
    results = {}
    for var, out_path in variants:
        print(f"\n==================================================")
        print(f"Exporting {var.upper()} -> {out_path}")
        print(f"==================================================")
        try:
            ckpt_to_use = checkpoint_path
            if var != "m6" and ("qat" in checkpoint_path.lower() or "m6" in checkpoint_path.lower()):
                base_ckpt = "pretrained/parseq_alpr_98.5.ckpt"
                if os.path.exists(base_ckpt):
                    ckpt_to_use = base_ckpt
            elif var == "m6":
                if "qat" in checkpoint_path.lower() or "m6" in checkpoint_path.lower():
                    ckpt_to_use = checkpoint_path
                elif os.path.exists("pretrained/parseq_alpr_qat_m6.ckpt"):
                    ckpt_to_use = "pretrained/parseq_alpr_qat_m6.ckpt"

            p = export_onnx(
                checkpoint_path=ckpt_to_use,
                variant=var,
                output_path=out_path,
                opset_version=opset,
                fuse_shapes=fuse_shapes,
                fuse_mha=fuse_mha,
                fuse_mlp=fuse_mlp,
                fuse_layernorm=fuse_layernorm,
            )
            results[var] = p
        except Exception as e:
            print(f"Error exporting {var}: {e}")
            results[var] = f"FAILED: {e}"
    print("\n=== Export Summary ===")
    for k, v in results.items():
        print(f"  {k.upper()}: {v}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="pretrained/parseq_alpr_98.5.ckpt")
    parser.add_argument("--variant", type=str, default="m1", choices=["m0", "m1", "m2", "m3", "m4", "m5", "m6"])
    parser.add_argument("--output", type=str, default="onnx/parseq_nar.onnx")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--all", action="store_true", help="Export all variants m0..m6")
    parser.add_argument("--fuse_shapes", action="store_true", help="Fuse redundant shape, reshape, cast and gather nodes via ONNX Simplifier")
    parser.add_argument("--fuse_mha", action="store_true", help="Emit canonical Softmax pattern allowing TensorRT FlashAttention/FMHA kernel fusion")
    parser.add_argument("--fuse_mlp", action="store_true", help="Emit canonical GELU allowing TensorRT FC1+GELU+FC2 GEMM kernel fusion")
    parser.add_argument("--fuse_layernorm", action="store_true", help="Emit canonical LayerNorm allowing TensorRT Myelin LayerNorm kernel fusion")
    parser.add_argument("--use_int_flashattention", action="store_true", help="Use INT-FlashAttention (arXiv:2409.16997v2)")
    parser.add_argument("--use_plugin", action="store_true", help="Use custom TensorRT plugins (INTFlashAttention, LayerNorm, GELU, Softmax)")
    parser.add_argument("--fusion_level", type=str, default="none", choices=["none", "shapes", "mha", "mlp", "all"],
                        help="Preset level of kernel fusion: none, shapes, mha (shapes+mha), mlp (shapes+mha+mlp), or all (full fusion)")
    args = parser.parse_args()

    fuse_shapes = args.fuse_shapes or (args.fusion_level in ["shapes", "mha", "mlp", "all"])
    fuse_mha = args.fuse_mha or (args.fusion_level in ["mha", "mlp", "all"])
    fuse_mlp = args.fuse_mlp or (args.fusion_level in ["mlp", "all"])
    fuse_layernorm = args.fuse_layernorm or (args.fusion_level in ["all"])

    if args.all:
        export_all_variants(
            checkpoint_path=args.checkpoint,
            opset=args.opset,
            fuse_shapes=fuse_shapes,
            fuse_mha=fuse_mha,
            fuse_mlp=fuse_mlp,
            fuse_layernorm=fuse_layernorm,
            use_int_flashattention=args.use_int_flashattention,
            use_plugin=args.use_plugin,
        )
    else:
        export_onnx(
            checkpoint_path=args.checkpoint,
            variant=args.variant,
            output_path=args.output,
            opset_version=args.opset,
            fuse_shapes=fuse_shapes,
            fuse_mha=fuse_mha,
            fuse_mlp=fuse_mlp,
            fuse_layernorm=fuse_layernorm,
            use_int_flashattention=args.use_int_flashattention,
            use_plugin=args.use_plugin,
        )
