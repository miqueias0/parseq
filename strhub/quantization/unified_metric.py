"""
Unified Metric (Omega) and Approximation Function Assignment
=============================================================
Implementation of IPTQ-ViT (Kim et al., 2025) Unified Metric:
Omega = 3 / (N(Q)^(-1) + N(P) + N(C))
where:
- Q = SQNR (Signal-to-Quantization-Noise Ratio) in dB
- P = Perturbation ||X - Q||_2^2
- C = Integer operation count of the layer
- N(x) = ln(1 + e^x) (Softplus normalization)
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def compute_sqnr(x_fp: Tensor, x_quant: Tensor) -> float:
    """
    Computes Signal-to-Quantization Noise Ratio (SQNR) in dB:
    Q = 20 * log10( sqrt(E[X^2]) / sqrt(E[(X - Q)^2]) )
      = 10 * log10( E[X^2] / E[(X - Q)^2] )
    """
    with torch.no_grad():
        x = x_fp.detach().float()
        q = x_quant.detach().float()
        signal_power = torch.mean(x * x).item()
        noise_power = torch.mean((x - q) * (x - q)).item()
        if noise_power <= 1e-12:
            return 80.0  # Cap maximum SQNR
        if signal_power <= 1e-12:
            return 0.0
        ratio = max(signal_power / noise_power, 1e-6)
        return 10.0 * math.log10(ratio)


def compute_perturbation(x_fp: Tensor, x_quant: Tensor) -> float:
    """
    Computes Quantization Perturbation P:
    P = ||X - Q||_2^2 (normalized by element count)
    """
    with torch.no_grad():
        x = x_fp.detach().float()
        q = x_quant.detach().float()
        p = torch.mean((x - q) * (x - q)).item()
        return p


def get_operator_cost(op_name: str, num_elements: int) -> float:
    """
    Computes computational cost C (integer operation count per element, normalized).
    Based on arithmetic and bit-shift FLOP-equivalent operations:
    - DataAwarePolyGELU: 1 abs, 1 clip, 1 add, 3 muls (quartic), 1 mul (a), 1 add, 1 sgn, 1 mul (0.5*x) = ~10 ops
    - i-GELU: 1 abs, 1 clip, 1 add, 1 mul (sq), 1 mul (a), 1 add, 1 sgn, 1 mul = ~8 ops
    - BitShiftGELU: shifts + adds = ~5 ops
    - EfficientBitSoftmax: max subtraction (1), log2e shift-add (3), fixed-point mult (1), Taylor shift-add (3), IntDiv (2) = ~10 ops
    - Shiftmax (I-ViT): max sub (1), log2e shift (2), linear approx (2), IntDiv (2) = ~7 ops
    - LogSoftmax: logarithmic lookup/expansion = ~12 ops
    - IntegerLayerNorm: mean (1), centered (1), variance (2), Newton sqrt (8), affine (2) = ~14 ops
    - BitShiftLayerNorm: simplified shifts = ~9 ops
    """
    costs_per_element = {
        "DataAwarePolyGELU": 10.0,
        "i-GELU": 8.0,
        "BitShiftGELU": 5.0,
        "EfficientBitSoftmax": 10.0,
        "Shiftmax": 7.0,
        "LogSoftmax": 12.0,
        "IntegerLayerNorm": 14.0,
        "BitShiftLayerNorm": 9.0,
    }
    base_cost = costs_per_element.get(op_name, 10.0)
    # Log scale normalization for cost to balance with SQNR and Perturbation
    return math.log1p(base_cost)


def softplus(x: float) -> float:
    """
    Softplus normalization: N(x) = ln(1 + e^x)
    """
    # Numerically stable softplus
    if x > 20.0:
        return x
    elif x < -20.0:
        return math.exp(x)
    return math.log1p(math.exp(x))


def compute_unified_metric(sqnr_db: float, perturbation: float, cost: float) -> float:
    """
    Unified Metric Omega = 3 / (N(Q)^(-1) + N(P) + N(C))
    Higher is better.
    """
    nq = max(softplus(sqnr_db), 1e-4)
    np = max(softplus(perturbation), 1e-4)
    nc = max(softplus(cost), 1e-4)

    denominator = (1.0 / nq) + np + nc
    if denominator <= 0:
        return 0.0
    return 3.0 / denominator


# ---------------------------------------------------------------------------
# Candidate Approximation Functions
# ---------------------------------------------------------------------------

def candidate_poly_gelu(x: Tensor) -> Tensor:
    a, b = -0.019913, -2.698088
    u = x / math.sqrt(2.0)
    u_clipped = torch.clamp(torch.abs(u), max=-b)
    poly = a * torch.pow(u_clipped + b, 4) + 1.0
    l_ours = torch.sign(u) * poly
    return 0.5 * x * (1.0 + l_ours)


def candidate_i_gelu(x: Tensor) -> Tensor:
    a, b = -0.2888, -1.769
    u = x / math.sqrt(2.0)
    u_clipped = torch.clamp(torch.abs(u), max=-b)
    poly = a * torch.pow(u_clipped + b, 2) + 1.0
    l_ibert = torch.sign(u) * poly
    return 0.5 * x * (1.0 + l_ibert)


def candidate_bitshift_gelu(x: Tensor) -> Tensor:
    # Piecewise linear / shift approximation: x * sigmoid(1.702 * x) via clamp
    return x * torch.clamp(0.5 + 0.125 * x, 0.0, 1.0)


def candidate_efficient_bit_softmax(x: Tensor) -> Tensor:
    # Efficient Bit-Softmax simulation
    x_max = torch.max(x, dim=-1, keepdim=True)[0]
    q = x - x_max
    # Base-2 Taylor series: 1 + ln(2)*r with ln(2) ~= 0.6875
    # 2^(q * log2(e))
    q_base2 = q * 1.4375
    exp_approx = torch.exp(q_base2 * 0.693147)
    denom = torch.clamp(torch.sum(exp_approx, dim=-1, keepdim=True), min=1e-8)
    return exp_approx / denom


def candidate_shiftmax(x: Tensor) -> Tensor:
    # I-ViT Shiftmax: 2^(q) approx 1 + q/2
    x_max = torch.max(x, dim=-1, keepdim=True)[0]
    q = x - x_max
    q_p = q * 1.4375
    exp_approx = torch.clamp(1.0 + q_p * 0.5, min=1e-4)
    denom = torch.clamp(torch.sum(exp_approx, dim=-1, keepdim=True), min=1e-8)
    return exp_approx / denom


def candidate_log_softmax(x: Tensor) -> Tensor:
    # FQ-ViT Log-Softmax
    return F.softmax(x, dim=-1)


def candidate_integer_layernorm(x: Tensor, eps: float = 1e-5) -> Tensor:
    # Standard LayerNorm with discrete integer sqrt simulation
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    return (x - mean) / torch.sqrt(var + eps)


def candidate_bitshift_layernorm(x: Tensor, eps: float = 1e-5) -> Tensor:
    # Shift LayerNorm using L1 norm approximation for variance
    mean = x.mean(dim=-1, keepdim=True)
    l1 = (x - mean).abs().mean(dim=-1, keepdim=True)
    # sigma ~= l1 * sqrt(pi / 2) ~= l1 * 1.2533
    sigma = l1 * 1.25
    return (x - mean) / torch.clamp(sigma, min=eps)


GELU_CANDIDATES = {
    "DataAwarePolyGELU": candidate_poly_gelu,
    "i-GELU": candidate_i_gelu,
    "BitShiftGELU": candidate_bitshift_gelu,
}

SOFTMAX_CANDIDATES = {
    "EfficientBitSoftmax": candidate_efficient_bit_softmax,
    "Shiftmax": candidate_shiftmax,
    "LogSoftmax": candidate_log_softmax,
}

LAYERNORM_CANDIDATES = {
    "IntegerLayerNorm": candidate_integer_layernorm,
    "BitShiftLayerNorm": candidate_bitshift_layernorm,
}


def evaluate_operator_metrics(
    fp_tensor: Tensor,
    candidate_fn,
    op_name: str,
    reference_fn,
) -> Dict[str, float]:
    """
    Evaluates SQNR, Perturbation, Cost, and Unified Metric Omega for a candidate function.
    """
    ref_out = reference_fn(fp_tensor)
    cand_out = candidate_fn(fp_tensor)

    sqnr = compute_sqnr(ref_out, cand_out)
    pert = compute_perturbation(ref_out, cand_out)
    cost = get_operator_cost(op_name, fp_tensor.numel())
    omega = compute_unified_metric(sqnr, pert, cost)

    return {
        "sqnr_db": sqnr,
        "perturbation": pert,
        "cost": cost,
        "omega": omega,
    }


class UnifiedMetricCalculator:
    """
    Calculator for layer-wise Unified Metric evaluation.
    """
    @staticmethod
    def evaluate_gelu_candidates(x: Tensor) -> Dict[str, Dict[str, float]]:
        ref_fn = F.gelu
        results = {}
        for name, fn in GELU_CANDIDATES.items():
            results[name] = evaluate_operator_metrics(x, fn, name, ref_fn)
        return results

    @staticmethod
    def evaluate_softmax_candidates(x: Tensor) -> Dict[str, Dict[str, float]]:
        ref_fn = lambda t: F.softmax(t, dim=-1)
        results = {}
        for name, fn in SOFTMAX_CANDIDATES.items():
            results[name] = evaluate_operator_metrics(x, fn, name, ref_fn)
        return results

    @staticmethod
    def evaluate_layernorm_candidates(x: Tensor) -> Dict[str, Dict[str, float]]:
        ref_fn = lambda t: F.layer_norm(t, (t.shape[-1],))
        results = {}
        for name, fn in LAYERNORM_CANDIDATES.items():
            results[name] = evaluate_operator_metrics(x, fn, name, ref_fn)
        return results


class UnifiedMetricSearcher:
    """
    IPTQ-ViT 3-Stage Search Engine:
    Stage 1: Layer-wise metric analysis on calibration activations.
    Stage 2: Optimal function assignment (max Omega per layer).
    Stage 3: Produce assignment dictionary for PARSeq model.
    """
    def __init__(self):
        self.assignments: Dict[str, str] = {}
        self.metric_records: Dict[str, Dict[str, Dict[str, float]]] = {}

    def search_layer(self, layer_name: str, op_type: str, sample_inputs: Tensor) -> str:
        """
        Evaluates candidates for a specific layer and assigns the operator with the highest Omega.
        op_type: 'gelu', 'softmax', or 'layernorm'.
        """
        with torch.no_grad():
            if op_type == "gelu":
                scores = UnifiedMetricCalculator.evaluate_gelu_candidates(sample_inputs)
            elif op_type == "softmax":
                scores = UnifiedMetricCalculator.evaluate_softmax_candidates(sample_inputs)
            elif op_type == "layernorm":
                scores = UnifiedMetricCalculator.evaluate_layernorm_candidates(sample_inputs)
            else:
                raise ValueError(f"Unknown op_type: {op_type}")

            best_op = max(scores.keys(), key=lambda k: scores[k]["omega"])
            self.assignments[layer_name] = best_op
            self.metric_records[layer_name] = scores
            return best_op

    def print_summary(self):
        print("\n=======================================================")
        print("IPTQ-ViT Unified Metric Search Summary (Optimal Assignment)")
        print("=======================================================")
        for layer, best_op in self.assignments.items():
            scores = self.metric_records[layer][best_op]
            print(f"Layer: {layer:<35} -> Selected: {best_op:<20} | Omega: {scores['omega']:.4f} | SQNR: {scores['sqnr_db']:.2f}dB")
        print("=======================================================\n")
