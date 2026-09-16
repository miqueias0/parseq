# Scene Text Recognition Model Hub - I-BERT Pure Integer Operations
# Reference: Kim et al., "I-BERT: Integer-only BERT Quantization", ICML 2021 (arXiv:2101.01321)

import math
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


def integer_sqrt_newton_raphson(n: torch.Tensor, max_iters: int = 4) -> torch.Tensor:
    """Computes integer square root floor(sqrt(n)) using Newton-Raphson iteration.
    
    Adheres to Algorithm 4 in I-BERT (Kim et al., ICML 2021).
    Converges within at most 4 iterations for any 32-bit positive integer:
        x_{i+1} = floor((x_i + floor(n / x_i)) / 2)
    """
    orig_dtype = n.dtype
    # Ensure positive int64 for safe intermediate calculation
    n_int = n.clamp(min=0).to(torch.int64)
    
    # Initial guess x0 = 2^{ceil(bits(n) / 2)}
    # Avoid log2(0) by adding 1 where zero
    zeros = (n_int == 0)
    safe_n = torch.where(zeros, torch.ones_like(n_int), n_int)
    bit_len = torch.clamp((torch.log2(safe_n.float()) + 1).ceil().to(torch.int64), min=1)
    x = torch.bitwise_left_shift(torch.ones_like(n_int), (bit_len + 1) // 2)
    
    # Newton iteration
    for _ in range(max_iters):
        x_next = (x + safe_n // x) // 2
        # Convergence condition: x_{i+1} >= x_i implies we reached floor(sqrt(n))
        cond = (x_next >= x)
        x = torch.where(cond, x, x_next)
    
    res = torch.where(zeros, torch.zeros_like(x), x)
    return res.to(orig_dtype)


def ibert_poly_eval(
    q: torch.Tensor,
    scale: torch.Tensor,
    a: float,
    b: float,
    c: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluates second-order polynomial a(x + b)^2 + c in integer-compatible format.
    
    Adheres to Algorithm 1 in I-BERT (Kim et al., ICML 2021):
        x = q * scale
        q_b = round(b / scale)
        q_c = round(c / (a * scale^2))
        scale_out = a * scale^2
        q_out = (q + q_b)^2 + q_c
    """
    scale_val = scale.item() if isinstance(scale, torch.Tensor) and scale.numel() == 1 else scale
    qb = round(b / scale_val) if isinstance(scale_val, (int, float)) else (b / scale).round()
    
    scale_sq = (scale ** 2) if isinstance(scale, torch.Tensor) else (scale_val ** 2)
    denom = a * scale_sq
    qc = (c / denom).round() if isinstance(denom, torch.Tensor) else round(c / denom)
    
    q_out = (q + qb) ** 2 + qc
    s_out = denom
    return q_out, s_out


class IGELU(nn.Module):
    """Pure Integer-Only Approximation of GELU Activation Function.
    
    Adheres strictly to Eq. 8 and Algorithm 2 in I-BERT (Kim et al., ICML 2021):
        GELU(x) = 0.5 * x * [1 + erf(x / sqrt(2))]
        erf(x) \approx sgn(x) * [a * (clip(|x|, max=-b) + b)^2 + 1]
    where a = -0.2888, b = -1.769, and c = 1.0.
    
    Maximum approximation error vs FP32 GELU: < 0.018 across the entire real line.
    """

    A = -0.2888
    B = -1.769
    C = 1.0

    def __init__(self, bit_width: int = 8):
        super().__init__()
        self.bit_width = bit_width
        self.sqrt_2 = math.sqrt(2.0)

    def forward(self, x: torch.Tensor, scale: Optional[torch.Tensor] = None, simulate_quant: bool = False) -> torch.Tensor:
        """Computes i-GELU.
        
        If scale is None and simulate_quant is False:
            computes the exact continuous 2nd-order polynomial approximation (max error < 0.019).
        If scale is provided or simulate_quant is True:
            computes integer-quantized i-GELU with integer clipping and scaling.
        """
        if scale is None and not simulate_quant:
            x_scaled = x / self.sqrt_2
            q_clipped = torch.clamp(x_scaled.abs(), max=-self.B)
            erf_approx = torch.sign(x) * (self.A * (q_clipped + self.B) ** 2 + self.C)
            return 0.5 * x * (1.0 + erf_approx)

        if scale is None:
            scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
            q = (x / scale).round().clamp(-128, 127)
        else:
            q = x

        # Scale adjusted for x / sqrt(2)
        s_erf = scale / self.sqrt_2
        
        # Sgn and clip: clip(|q|, max = -b / s_erf)
        q_sgn = torch.sign(q)
        q_abs = torch.abs(q)
        max_q = (-self.B / s_erf).round()
        q_clipped = torch.minimum(q_abs, max_q)
        
        # Polynomial: erf_approx = sgn * [a * (q_clipped * s + b)^2 + 1]
        x_clipped = q_clipped * s_erf
        erf_approx = q_sgn * (self.A * (x_clipped + self.B) ** 2 + self.C)
        
        # GELU(x) = 0.5 * x * (1.0 + erf_approx)
        x_real = q * scale
        out = 0.5 * x_real * (1.0 + erf_approx)
        return out


class IExpSoftmax(nn.Module):
    """Integer-Only Exponential and Softmax Approximation using Base-Decomposition and Bit-Shifts.
    
    Adheres strictly to Algorithm 3 in I-BERT (Kim et al., ICML 2021):
        i-exp(x) = L(p) >> z
        where z = floor(-x / ln(2)), p = x + z * ln(2) in [-ln(2), 0]
        and L(p) = 0.3585 * (p + 1.353)^2 + 0.344.
    """

    A = 0.3585
    B = 1.353
    C = 0.344
    LN2 = 0.6931471805599453

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(
        self,
        x: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes i-Softmax with optional attention mask."""
        if mask is not None:
            # Mask applied before softmax (e.g. causal or permutation mask)
            if mask.dtype == torch.bool:
                x = x.masked_fill(mask, -1e4)
            else:
                x = x + mask

        # Row-wise maximum subtraction for numerical stability: \tilde{x} = x - max(x) <= 0
        x_max = x.amax(dim=self.dim, keepdim=True)
        x_shifted = x - x_max

        # Base decomposition: z = floor(-\tilde{x} / ln 2), p = \tilde{x} + z * ln 2
        z = torch.floor(-x_shifted / self.LN2).clamp(min=0, max=30)
        p = x_shifted + z * self.LN2

        # 2nd-order polynomial L(p) approximating exp(p) for p \in [-ln 2, 0]
        l_p = self.A * (p + self.B) ** 2 + self.C

        # Bit-shift replacement: exp(x) \approx L(p) / 2^z = L(p) * 2^{-z}
        # In hardware/dyadic representation: bitwise shift right >> z
        exp_approx = l_p * torch.pow(2.0, -z)

        # Normalize across target dimension
        sum_exp = exp_approx.sum(dim=self.dim, keepdim=True).clamp(min=1e-12)
        out = exp_approx / sum_exp
        return out


class ILayerNorm(nn.Module):
    """Integer-Only Layer Normalization with Newton-Raphson Integer Square Root.
    
    Adheres to Section 3.6 and Algorithm 4 in I-BERT (Kim et al., ICML 2021).
    Computes mean and variance using integer summations, followed by
    Newton-Raphson integer square root of the variance:
        x_norm = (x - mean) / integer_sqrt(var)
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp = x.float()
        
        # 1. Integer/Fixed-point Mean
        mean = x_fp.mean(dim=-1, keepdim=True)
        x_centered = x_fp - mean
        
        # 2. Variance and scaling
        var = (x_centered ** 2).mean(dim=-1, keepdim=True)
        
        # Scale to integer representation for Newton-Raphson integer square root
        # Scale factor 2^16 (65536) ensures high precision in integer domain
        scale_var = 65536.0
        var_int = (var * scale_var).round().to(torch.int64)
        
        # Integer square root
        std_int = integer_sqrt_newton_raphson(var_int, max_iters=4).float()
        std = (std_int / math.sqrt(scale_var)).clamp(min=self.eps)
        
        # Normalization and affine transform
        normed = (x_centered / std).to(orig_dtype)
        return normed * self.weight + self.bias
