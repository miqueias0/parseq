import os
import sys
import json
import argparse
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import onnxruntime as ort

from strhub.models.utils import load_from_checkpoint
from strhub.models.parseq.quantized_parseq import create_model_variant
from strhub.quant.quant_utils import compute_sqnr, compute_mse, compute_cosine_similarity


def validate_onnx_model(
    checkpoint_path: str,
    onnx_path: str,
    variant: str = "m1",
    num_samples: int = 20,
    img_size: tuple = (32, 128)
) -> Dict[str, Any]:
    device = torch.device("cpu")
    system = load_from_checkpoint(checkpoint_path).eval()
    model = create_model_variant(variant, system.model).eval().to(device)

    # Initialize ONNX Runtime session
    providers = ["CPUExecutionProvider"]
    if ort.get_device() == "GPU" and "CUDAExecutionProvider" in ort.get_available_providers():
        providers.insert(0, "CUDAExecutionProvider")

    session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name

    max_abs_diffs = []
    mses = []
    cos_sims = []
    exact_agreements = 0

    base = model.model if hasattr(model, "model") else model
    num_steps = base.max_label_length + 1

    for _ in range(num_samples):
        dummy_input = torch.randn(1, 3, img_size[0], img_size[1], dtype=torch.float32)

        # PyTorch forward
        with torch.no_grad():
            memory = base.encode(dummy_input)
            batch_dummy = torch.zeros_like(dummy_input[:, :1, 0, 0]).unsqueeze(-1)
            pos_queries = base.pos_queries[:, :num_steps].to(dummy_input.device) + batch_dummy
            tgt_in = torch.full((1, 1), system.tokenizer.bos_id, dtype=torch.long, device=dummy_input.device)
            tgt_out = base.decode(tgt_in, memory, tgt_query=pos_queries)
            pt_logits = base.head(tgt_out)

        # ONNX Runtime forward
        ort_inputs = {input_name: dummy_input.numpy()}
        ort_logits = session.run(None, ort_inputs)[0]
        ort_logits_t = torch.from_numpy(ort_logits)

        # Numerical comparison
        diff = torch.abs(pt_logits - ort_logits_t)
        max_abs_diffs.append(float(diff.max().item()))
        mses.append(compute_mse(pt_logits, ort_logits_t))
        cos_sims.append(compute_cosine_similarity(pt_logits, ort_logits_t))

        # Prediction comparison
        pt_preds, _ = system.tokenizer.decode(pt_logits.softmax(-1))
        ort_preds, _ = system.tokenizer.decode(ort_logits_t.softmax(-1))
        if pt_preds == ort_preds:
            exact_agreements += 1

    mean_max_diff = float(np.mean(max_abs_diffs))
    mean_mse = float(np.mean(mses))
    mean_cos = float(np.mean(cos_sims))
    agreement_pct = float(exact_agreements / num_samples * 100.0)

    print(f"=== ONNX Validation Results ({variant.upper()}) ===")
    print(f"Mean Max Absolute Difference: {mean_max_diff:.6f}")
    print(f"Mean MSE: {mean_mse:.6e}")
    print(f"Mean Cosine Similarity: {mean_cos:.6f}")
    print(f"Prediction Agreement: {agreement_pct:.1f}% ({exact_agreements}/{num_samples})")

    status = "PASSED" if mean_cos > 0.99 and agreement_pct >= 90.0 else "WARNING"
    print(f"ONNX Validation Status: {status}")

    return {
        "variant": variant,
        "onnx_path": onnx_path,
        "mean_max_abs_diff": mean_max_diff,
        "mean_mse": mean_mse,
        "mean_cosine_similarity": mean_cos,
        "prediction_agreement_pct": agreement_pct,
        "status": status,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="pretrained/parseq_alpr_98.5.ckpt")
    parser.add_argument("--onnx", type=str, default="onnx/parseq_nar.onnx")
    parser.add_argument("--variant", type=str, default="m1")
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()

    validate_onnx_model(
        checkpoint_path=args.checkpoint,
        onnx_path=args.onnx,
        variant=args.variant,
        num_samples=args.samples
    )
