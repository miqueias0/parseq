import math
from typing import Dict, Any, Tuple, List
import torch
import torch.nn.functional as F
from .quant_utils import compute_sqnr, compute_mse


# Operation count estimation (normalized integer operations/FLOPs per element)
# Based on IPTQ-ViT Section 4.5
COMPUTATIONAL_COSTS = {
    # GELU candidates
    "gelu_fp32": 15,
    "gelu_ibert": 6,       # second-order polynomial + clip
    "gelu_ivit": 4,        # shift-based
    "gelu_iptq": 8,        # quartic polynomial with term reuse
    # Softmax candidates
    "softmax_fp32": 25,
    "softmax_ibert": 14,   # max subtract, range reduction, poly, shift
    "softmax_ivit": 8,     # Shiftmax
    "softmax_iptq": 10,    # Efficient Bit-exp (first degree Taylor + shift) + IntDiv
    # LayerNorm candidates
    "layernorm_fp32": 20,
    "layernorm_ibert": 12, # mean, var, 4-iter Newton integer sqrt, affine
    "layernorm_iptq": 10,
}


def softplus(x: float) -> float:
    """Softplus function N(x) = log(1 + exp(x))."""
    # Guard against overflow
    if x > 20.0:
        return x
    if x < -20.0:
        return math.exp(x)
    return math.log(1.0 + math.exp(x))


def compute_unified_metric(Q: float, P: float, C: float) -> float:
    """Compute the Unified Metric (Omega) proposed in IPTQ-ViT (CVPR 2024, Eq. 16).
    Omega = 3 / (N(Q)^(-1) + N(P) + N(C))
    where:
        Q: Quantization sensitivity (SQNR in dB, higher is better)
        P: Quantization perturbation (MSE, lower is better)
        C: Computational cost (operation count, lower is better)
    """
    n_q = softplus(Q)
    n_p = softplus(P)
    n_c = softplus(C)

    inv_n_q = 1.0 / max(n_q, 1e-8)
    denom = inv_n_q + n_p + n_c
    omega = 3.0 / max(denom, 1e-8)
    return omega


def evaluate_layer_candidates(
    layer_name: str,
    layer_type: str,
    input_tensor: torch.Tensor,
    candidates: Dict[str, torch.nn.Module]
) -> Tuple[str, Dict[str, Any]]:
    """Evaluate all candidate approximation functions for a layer and assign the best candidate
    maximizing the Unified Metric Omega.
    """
    with torch.no_grad():
        fp_reference = candidates[f"{layer_type}_fp32"](input_tensor)
        scores = {}
        best_candidate = None
        best_omega = -float("inf")

        for name, candidate_module in candidates.items():
            approx_output = candidate_module(input_tensor)
            q = compute_sqnr(fp_reference, approx_output)
            p = compute_mse(fp_reference, approx_output)
            c = float(COMPUTATIONAL_COSTS.get(name, 10))
            omega = compute_unified_metric(q, p, c)

            scores[name] = {
                "Q_SQNR_dB": q,
                "P_MSE": p,
                "C_cost": c,
                "Unified_Metric_Omega": omega,
            }

            if omega > best_omega:
                best_omega = omega
                best_candidate = name

        return best_candidate, {
            "layer_name": layer_name,
            "layer_type": layer_type,
            "best_candidate": best_candidate,
            "scores": scores,
        }
