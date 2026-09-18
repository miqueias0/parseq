import os
import sys
import json
import csv
import argparse
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint
from strhub.quant.quant_utils import (
    quantize_symmetric,
    dequantize_symmetric,
    compute_sqnr,
    compute_mse,
    compute_cosine_similarity,
    compute_saturation_stats,
    compute_distribution_stats,
)
from strhub.quant.integer_gelu import IBERTGELU, IViTGELU, IPTQDataAwarePolyGELU, GELUFP32
from strhub.quant.integer_softmax import IBERTSoftmax, IViTShiftmax, IPTQBitSoftmax, SoftmaxFP32
from strhub.quant.integer_layernorm import IBERTLayerNorm, IPTQLayerNorm, LayerNormFP32
from strhub.quant.unified_metric import evaluate_layer_candidates, compute_unified_metric


def run_layer_sensitivity_analysis(
    checkpoint_path: str,
    dataset_name: str = "VeSV_pad",
    data_dir: str = "data",
    num_samples: int = 128,
    output_dir: str = "results/sensitivity"
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    system = load_from_checkpoint(checkpoint_path).eval()
    base_model = system.model.to(device)

    data_module = SceneTextDataModule(
        root_dir=data_dir,
        train_dir=dataset_name,
        img_size=system.hparams.img_size,
        max_label_length=system.hparams.max_label_length,
        charset_train=system.hparams.charset_train,
        charset_test=system.hparams.charset_test,
        batch_size=32,
        num_workers=0,
        augment=False
    )
    dataset = data_module.train_dataset
    sub_indices = list(range(min(num_samples, len(dataset))))
    loader = DataLoader(Subset(dataset, sub_indices), batch_size=32, shuffle=False)

    # Collect sample activations
    sample_images = []
    for imgs, _ in loader:
        sample_images.append(imgs.to(device))
        if sum(x.shape[0] for x in sample_images) >= num_samples:
            break
    calib_batch = torch.cat(sample_images, dim=0)[:num_samples]

    layer_reports = []
    assignment = {}

    # 1. Evaluate Encoder blocks
    for i, block in enumerate(base_model.encoder.blocks):
        # Extract activations before MLP activation (GELU)
        # Standard ViT MLP: fc1 -> act -> fc2
        with torch.no_grad():
            # Run forward up to this block to get input
            # Hook the mlp act input
            captured_act = []
            def gelu_hook(m, inp, out):
                captured_act.append(inp[0].detach())

            h = block.mlp.act.register_forward_hook(gelu_hook)
            _ = base_model(system.tokenizer, calib_batch)
            h.remove()

            if captured_act:
                act_tensor = captured_act[0]
                gelu_candidates = {
                    "gelu_fp32": GELUFP32().to(device),
                    "gelu_ibert": IBERTGELU().to(device),
                    "gelu_ivit": IViTGELU().to(device),
                    "gelu_iptq": IPTQDataAwarePolyGELU().to(device),
                }
                best_gelu, gelu_eval = evaluate_layer_candidates(
                    layer_name=f"encoder.block_{i}.gelu",
                    layer_type="gelu",
                    input_tensor=act_tensor,
                    candidates=gelu_candidates
                )
                assignment[f"encoder.block_{i}.gelu"] = best_gelu
                layer_reports.append({
                    "layer": f"encoder.block_{i}.gelu",
                    "type": "GELU",
                    "best_approximation": best_gelu,
                    "SQNR_dB": gelu_eval["scores"][best_gelu]["Q_SQNR_dB"],
                    "MSE": gelu_eval["scores"][best_gelu]["P_MSE"],
                    "Unified_Metric": gelu_eval["scores"][best_gelu]["Unified_Metric_Omega"],
                })

            # Hook LayerNorm 1 input
            captured_ln = []
            def ln_hook(m, inp, out):
                captured_ln.append(inp[0].detach())

            h_ln = block.norm1.register_forward_hook(ln_hook)
            _ = base_model(system.tokenizer, calib_batch)
            h_ln.remove()

            if captured_ln:
                ln_tensor = captured_ln[0]
                ln_candidates = {
                    "layernorm_fp32": LayerNormFP32(block.norm1.normalized_shape[0]).to(device),
                    "layernorm_ibert": IBERTLayerNorm(block.norm1.normalized_shape[0]).to(device),
                    "layernorm_iptq": IPTQLayerNorm(block.norm1.normalized_shape[0]).to(device),
                }
                best_ln, ln_eval = evaluate_layer_candidates(
                    layer_name=f"encoder.block_{i}.layernorm",
                    layer_type="layernorm",
                    input_tensor=ln_tensor,
                    candidates=ln_candidates
                )
                assignment[f"encoder.block_{i}.layernorm"] = best_ln
                layer_reports.append({
                    "layer": f"encoder.block_{i}.norm1",
                    "type": "LayerNorm",
                    "best_approximation": best_ln,
                    "SQNR_dB": ln_eval["scores"][best_ln]["Q_SQNR_dB"],
                    "MSE": ln_eval["scores"][best_ln]["P_MSE"],
                    "Unified_Metric": ln_eval["scores"][best_ln]["Unified_Metric_Omega"],
                })

    # 2. Evaluate Decoder layers
    for i, layer in enumerate(base_model.decoder.layers):
        with torch.no_grad():
            captured_dec_gelu = []
            def dec_gelu_hook(m, inp, out):
                captured_dec_gelu.append(out.detach())

            h_dec = layer.linear1.register_forward_hook(dec_gelu_hook)
            _ = base_model(system.tokenizer, calib_batch)
            h_dec.remove()

            if captured_dec_gelu:
                act_dec = captured_dec_gelu[0]
                gelu_candidates = {
                    "gelu_fp32": GELUFP32().to(device),
                    "gelu_ibert": IBERTGELU().to(device),
                    "gelu_ivit": IViTGELU().to(device),
                    "gelu_iptq": IPTQDataAwarePolyGELU().to(device),
                }
                best_dec_gelu, dec_gelu_eval = evaluate_layer_candidates(
                    layer_name=f"decoder.layer_{i}.gelu",
                    layer_type="gelu",
                    input_tensor=act_dec,
                    candidates=gelu_candidates
                )
                assignment[f"decoder.layer_{i}.gelu"] = best_dec_gelu
                layer_reports.append({
                    "layer": f"decoder.layer_{i}.gelu",
                    "type": "GELU",
                    "best_approximation": best_dec_gelu,
                    "SQNR_dB": dec_gelu_eval["scores"][best_dec_gelu]["Q_SQNR_dB"],
                    "MSE": dec_gelu_eval["scores"][best_dec_gelu]["P_MSE"],
                    "Unified_Metric": dec_gelu_eval["scores"][best_dec_gelu]["Unified_Metric_Omega"],
                })

    # Save assignments JSON
    json_path = os.path.join(output_dir, "approximation_assignment.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(assignment, f, indent=2)

    # Save CSV Report
    csv_path = os.path.join(output_dir, "layer_sensitivity_report.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        if layer_reports:
            writer = csv.DictWriter(f, fieldnames=list(layer_reports[0].keys()))
            writer.writeheader()
            writer.writerows(layer_reports)

    print(f"Layer sensitivity analysis complete. Evaluated {len(layer_reports)} non-linear layers.")
    print(f"Saved assignment to {json_path} and CSV report to {csv_path}.")
    return {"assignment": assignment, "reports": layer_reports}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="pretrained/parseq_alpr_98.5.ckpt")
    parser.add_argument("--dataset", type=str, default="VeSV_pad")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="results/sensitivity")
    args = parser.parse_args()

    run_layer_sensitivity_analysis(
        checkpoint_path=args.checkpoint,
        dataset_name=args.dataset,
        num_samples=args.samples,
        output_dir=args.output_dir
    )
