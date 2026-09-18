import math
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def quantize_symmetric(
    x: torch.Tensor,
    bits: int = 8,
    scale: Optional[torch.Tensor] = None,
    channel_wise: bool = False,
    dim: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Uniform symmetric quantization to [-2^(b-1), 2^(b-1) - 1].
    For 8 bits: [-128, 127].
    """
    qmin = -(1 << (bits - 1))
    qmax = (1 << (bits - 1)) - 1

    if scale is None:
        if channel_wise:
            max_val = x.abs().amax(dim=[i for i in range(x.dim()) if i != dim], keepdim=True)
        else:
            max_val = x.abs().max()
        scale = torch.clamp(max_val / qmax, min=1e-8)

    q = torch.clamp(torch.round(x / scale), qmin, qmax)
    return q, scale


def dequantize_symmetric(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize: x_approx = q * scale."""
    return q * scale


def quantize_naive(x: torch.Tensor, bits: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """Naive INT8: Direct truncation/clamping to [-128, 127] without proper dynamic range adaptation.
    Used explicitly as a NEGATIVE CONTROL to show degradation without calibration.
    """
    qmin = -(1 << (bits - 1))
    qmax = (1 << (bits - 1)) - 1
    # Directly round float values and clamp into int8 range
    q = torch.clamp(torch.trunc(x), qmin, qmax)
    scale = torch.tensor(1.0, device=x.device, dtype=x.dtype)
    return q, scale


def compute_sqnr(original: torch.Tensor, quantized: torch.Tensor, eps: float = 1e-10) -> float:
    """Signal-to-Quantization-Noise Ratio (SQNR in dB).
    SQNR = 10 * log10(E[X^2] / E[(X - Q)^2]) or 20 * log10(||X|| / ||X - Q||).
    Following IPTQ-ViT Eq. 18: Q = 20 * log10(sqrt(E[X^2]) / sqrt(E[(X - Q)^2])) = 10 * log10(...)
    """
    signal_power = torch.mean(original.float() ** 2)
    noise_power = torch.mean((original.float() - quantized.float()) ** 2)
    if noise_power < eps:
        return 100.0  # Cap perfect reconstruction
    sqnr = 10.0 * torch.log10(signal_power / (noise_power + eps))
    return float(sqnr.item())


def compute_mse(original: torch.Tensor, quantized: torch.Tensor) -> float:
    """Mean Squared Error (L2 perturbation / MSE)."""
    return float(torch.mean((original.float() - quantized.float()) ** 2).item())


def compute_cosine_similarity(original: torch.Tensor, quantized: torch.Tensor) -> float:
    """Cosine similarity between flattened original and quantized representations."""
    flat_orig = original.float().flatten().unsqueeze(0)
    flat_quant = quantized.float().flatten().unsqueeze(0)
    cos = F.cosine_similarity(flat_orig, flat_quant, dim=1)
    return float(cos.item())


def compute_l2_error(original: torch.Tensor, approx: torch.Tensor) -> float:
    """Relative or normalized L2 error ||x - approx||_2 / ||x||_2."""
    diff_norm = torch.norm(original.float() - approx.float(), p=2)
    orig_norm = torch.norm(original.float(), p=2) + 1e-8
    return float((diff_norm / orig_norm).item())


def compute_linf_error(original: torch.Tensor, approx: torch.Tensor) -> float:
    """L-infinity max absolute error max(|x - approx|)."""
    return float(torch.max(torch.abs(original.float() - approx.float())).item())


def compute_saturation_stats(q_tensor: torch.Tensor, bits: int = 8) -> Dict[str, float]:
    """Compute saturation percentage at min and max integer limits."""
    qmin = -(1 << (bits - 1))
    qmax = (1 << (bits - 1)) - 1
    total_elements = q_tensor.numel()
    if total_elements == 0:
        return {"min_saturated_pct": 0.0, "max_saturated_pct": 0.0, "total_saturated_pct": 0.0}

    min_count = torch.sum(q_tensor <= qmin).item()
    max_count = torch.sum(q_tensor >= qmax).item()
    return {
        "min_saturated_pct": float(min_count / total_elements * 100.0),
        "max_saturated_pct": float(max_count / total_elements * 100.0),
        "total_saturated_pct": float((min_count + max_count) / total_elements * 100.0),
    }


def compute_distribution_stats(tensor: torch.Tensor) -> Dict[str, float]:
    """Compute distribution statistics for outlier analysis."""
    flat = tensor.float().flatten()
    if flat.numel() == 0:
        return {}

    mean = float(flat.mean().item())
    std = float(flat.std().item())
    min_val = float(flat.min().item())
    max_val = float(flat.max().item())

    # Fast percentile estimation with subsampling if large
    if flat.numel() > 10000:
        step = flat.numel() // 10000
        sample_flat = flat[::step]
    else:
        sample_flat = flat

    # Percentiles using torch.quantile
    q_levels = torch.tensor([0.01, 0.05, 0.50, 0.95, 0.99, 0.999], device=sample_flat.device)
    quantiles = torch.quantile(sample_flat, q_levels).tolist()

    return {
        "min": min_val,
        "max": max_val,
        "mean": mean,
        "std": std,
        "p1": quantiles[0],
        "p5": quantiles[1],
        "p50": quantiles[2],
        "p95": quantiles[3],
        "p99": quantiles[4],
        "p99_9": quantiles[5],
    }
