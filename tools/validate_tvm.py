#!/usr/bin/env python3
"""Numerical Audit and Validation Tool for Apache TVM Models.
Performs tensor-to-tensor comparison against PyTorch FP32 reference baseline:
- Max Absolute Difference, MSE, and Cosine Similarity
- Exact Plate Prediction Agreement (%)
- Detailed Divergence Character Index Diagnostics
- Verification Status (PASSED / WARNING)
Mirrors architectural parity with tools/validate_tensorrt.py.
"""

import os
import sys

# Ensure UTF-8 output encoding across platforms
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

os.environ["TVM_FFI_DISABLE_TORCH_C_DLPACK"] = "1"

import json
import argparse
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from strhub.models.utils import load_from_checkpoint
from strhub.models.tvm_utils import TVMRuntimeSession
from strhub.quant.quant_utils import compute_sqnr, compute_mse, compute_cosine_similarity


def validate_tvm_library(
    lib_path: str,
    checkpoint_path: str = "pretrained/parseq_alpr_98.5.ckpt",
    num_samples: int = 20,
    tolerance_cossim: float = 0.99,
    img_size: tuple = (32, 128),
    device: str = "cuda"
) -> Dict[str, Any]:
    """Validates an Apache TVM library (.so / .tar) against PyTorch FP32 model predictions."""
    if not os.path.isfile(lib_path):
        raise FileNotFoundError(f"Biblioteca TVM não encontrada: {lib_path}")

    # Fallback to CPU if CUDA is requested but not available
    dev_str = device.lower()
    if "cuda" in dev_str and not torch.cuda.is_available():
        dev_str = "cpu"

    pt_device = torch.device(dev_str)
    session = TVMRuntimeSession(lib_path, device=dev_str)

    system = load_from_checkpoint(checkpoint_path).eval().to(pt_device)
    base = system.model
    num_steps = base.max_label_length + 1

    max_abs_diffs = []
    mses = []
    cos_sims = []
    exact_agreements = 0
    divergence_diagnostics = []

    print(f"\n==================================================")
    print(f"Iniciando Validação Numérica TVM: {os.path.basename(lib_path)}")
    print(f"Referência PyTorch: {checkpoint_path} | Amostras: {num_samples} | Tolerância: {tolerance_cossim}")
    print(f"==================================================")

    for idx in range(num_samples):
        dummy_input = torch.randn(1, 3, img_size[0], img_size[1], dtype=torch.float32, device=pt_device)

        # 1. Forward PyTorch FP32 de referência
        with torch.no_grad():
            memory = base.encode(dummy_input)
            pos_queries = base.pos_queries[:, :num_steps].expand(1, -1, -1)
            tgt_in = torch.full((1, 1), system.tokenizer.bos_id, dtype=torch.long, device=base._device)
            tgt_out = base.decode(tgt_in, memory, tgt_query=pos_queries)
            pt_logits = base.head(tgt_out).cpu()

        # 2. Forward Apache TVM
        inp_np = dummy_input.cpu().numpy()
        tvm_logits_np = session.run(inp_np)
        tvm_logits_t = torch.from_numpy(tvm_logits_np).cpu()

        # 3. Métricas de divergência numérica
        diff = torch.abs(pt_logits - tvm_logits_t)
        max_abs_diffs.append(float(diff.max().item()))
        mses.append(compute_mse(pt_logits, tvm_logits_t))
        cos_sims.append(compute_cosine_similarity(pt_logits, tvm_logits_t))

        # 4. Decodificação de caracteres e concordância
        pt_preds, _ = system.tokenizer.decode(pt_logits.softmax(-1))
        tvm_preds, _ = system.tokenizer.decode(tvm_logits_t.softmax(-1))

        if pt_preds == tvm_preds:
            exact_agreements += 1
        else:
            pred_pt = pt_preds[0] if pt_preds else ""
            pred_tvm = tvm_preds[0] if tvm_preds else ""
            divergence_idx = -1
            for ci, (cp, ct) in enumerate(zip(pred_pt, pred_tvm)):
                if cp != ct:
                    divergence_idx = ci
                    break
            divergence_diagnostics.append({
                "sample_idx": idx,
                "pytorch_pred": pred_pt,
                "tvm_pred": pred_tvm,
                "first_divergence_char_idx": divergence_idx,
            })

    mean_max_diff = float(np.mean(max_abs_diffs))
    mean_mse = float(np.mean(mses))
    mean_cos = float(np.mean(cos_sims))
    agreement_pct = float(exact_agreements / num_samples * 100.0)

    # 5. Emissão de status
    status = "PASSED" if (mean_cos >= tolerance_cossim and agreement_pct >= 95.0) else "WARNING"

    print(f"\n=== Resultados de Validação TVM ===")
    print(f"Mean Max Absolute Difference: {mean_max_diff:.6f}")
    print(f"Mean MSE: {mean_mse:.6e}")
    print(f"Mean Cosine Similarity: {mean_cos:.6f}")
    print(f"Prediction Agreement: {agreement_pct:.1f}% ({exact_agreements}/{num_samples})")
    print(f"Status da Auditoria: {status}")
    if divergence_diagnostics:
        print(f"Divergências detectadas ({len(divergence_diagnostics)} amostras): {divergence_diagnostics[:2]}")

    return {
        "lib_path": lib_path,
        "checkpoint_path": checkpoint_path,
        "num_samples": num_samples,
        "mean_max_abs_diff": mean_max_diff,
        "mean_mse": mean_mse,
        "mean_cosine_similarity": mean_cos,
        "prediction_agreement_pct": agreement_pct,
        "status": status,
        "divergences": divergence_diagnostics,
    }


def evaluate_tvm_dataset(
    lib_path: str,
    data_loader,
    tokenizer,
    charset_adapter,
    max_samples: int = 200,
    device: str = "cuda"
) -> Dict[str, Any]:
    """Avalia acurácia e NED diretamente em um DataLoader do PyTorch usando a biblioteca TVM."""
    from nltk import edit_distance
    dev_str = device.lower()
    if "cuda" in dev_str and not torch.cuda.is_available():
        dev_str = "cpu"

    session = TVMRuntimeSession(lib_path, device=dev_str)

    exact_matches = []
    ned_scores = []
    cer_scores = []
    total_evaluated = 0

    for images, labels in data_loader:
        if max_samples and total_evaluated >= max_samples:
            break
        for img, gt in zip(images, labels):
            img_np = img.unsqueeze(0).numpy()
            out_np = session.run(img_np)
            logits_t = torch.from_numpy(out_np)
            probs = logits_t.softmax(-1)
            preds, _ = tokenizer.decode(probs)
            pred = charset_adapter(preds[0]) if preds else ""

            is_exact = 1.0 if pred == gt else 0.0
            exact_matches.append(is_exact)
            ed = edit_distance(pred, gt)
            max_l = max(len(pred), len(gt), 1)
            ned_scores.append(1.0 - (ed / max_l))
            cer_scores.append(ed / max(len(gt), 1))

            total_evaluated += 1
            if max_samples and total_evaluated >= max_samples:
                break

    acc = float(np.mean(exact_matches)) if exact_matches else 0.0
    ned = float(np.mean(ned_scores)) if ned_scores else 0.0
    cer = float(np.mean(cer_scores)) if cer_scores else 0.0

    return {
        "num_samples": total_evaluated,
        "exact_plate_accuracy": acc,
        "normalized_edit_distance": ned,
        "character_error_rate": cer,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Numerical Validation for Apache TVM Models")
    parser.add_argument("--lib", type=str, required=True, help="Caminho da biblioteca TVM compilada (.so / .tar)")
    parser.add_argument("--base_checkpoint", default="pretrained/parseq_alpr_98.5.ckpt", help="Checkpoint base PyTorch FP32")
    parser.add_argument("--samples", type=int, default=20, help="Número de amostras aleatórias para validação")
    parser.add_argument("--tolerance_cossim", type=float, default=0.99, help="Tolerância mínima de Cosine Similarity")
    parser.add_argument("--device", type=str, default="cuda", help="Dispositivo de execução (cuda / cpu)")
    parser.add_argument("--output", type=str, default=None, help="Caminho para salvar o relatório JSON")
    args = parser.parse_args()

    res = validate_tvm_library(
        lib_path=args.lib,
        checkpoint_path=args.base_checkpoint,
        num_samples=args.samples,
        tolerance_cossim=args.tolerance_cossim,
        device=args.device,
    )

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"Relatório de validação salvo em: {args.output}")
