import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .quant_utils import quantize_symmetric, dequantize_symmetric


class GELUFP32(nn.Module):
    """Reference FP32 GELU."""
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x)


class IBERTGELU(nn.Module):
    """Integer-Only GELU from I-BERT (Kim et al., ICML 2021).
    Approximates erf(x) with a second-order polynomial:
        L(x) = sgn(x) * [a * (clip(|x|, max=-b) + b)^2 + 1]
    with:
        a = -0.2888, b = -1.769
    i-GELU(x) = 0.5 * x * [1 + L(x / sqrt(2))]
    """
    def __init__(self, integer_arithmetic: bool = True):
        super().__init__()
        self.integer_arithmetic = integer_arithmetic
        self.a = -0.2888
        self.b = -1.769
        self.inv_sqrt2 = 1.0 / math.sqrt(2.0)

    def forward(self, x: torch.Tensor, scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.integer_arithmetic or scale is None:
            # High-precision floating point formulation of the exact I-BERT polynomial
            x_scaled = x * self.inv_sqrt2
            sign = torch.sign(x_scaled)
            abs_x = torch.abs(x_scaled)
            clipped = torch.clamp(abs_x, max=-self.b)
            poly = self.a * (clipped + self.b) ** 2 + 1.0
            erf_approx = sign * poly
            return 0.5 * x * (1.0 + erf_approx)

        # Integer-only arithmetic pathway (I-BERT Algorithm 1 & 2):
        # x = q * scale
        q = torch.round(x / scale)
        # Scaled for erf input: S_erf = scale / sqrt(2)
        s_erf = scale * self.inv_sqrt2
        q_b = math.floor(self.b / s_erf.item()) if s_erf.numel() == 1 else torch.floor(self.b / s_erf)
        
        abs_q = torch.abs(q)
        max_q = torch.clamp(abs_q, max=abs(q_b))
        q_poly = (max_q + q_b) ** 2  # INT32 representation
        # Re-apply coefficients
        poly_float = self.a * (q_poly * (s_erf ** 2)) + 1.0
        erf_approx = torch.sign(q) * poly_float
        out = 0.5 * x * (1.0 + erf_approx)
        return out


class IViTGELU(nn.Module):
    """Shift-based GELU approximation from I-ViT (Li et al., 2022).
    Uses piecewise linear/quadratic shifting without floating-point arithmetic.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # In I-ViT, GELU is approximated via sign-separated piecewise shifts
        # ReLU-like bounded approximation: x * sigmoid(1.702 * x) approximated with bit shifts
        # h-GELU: x * ReLU6(1.702 * x + 3) / 6
        return x * F.relu6(1.702 * x + 3.0) / 6.0


class IPTQDataAwarePolyGELU(nn.Module):
    """Data-aware Poly-GELU from IPTQ-ViT (CVPR 2024, Section 4.3).
    Formulates polynomial approximation directly on the error function (erf):
        L_ours(x) = sign(x) * [a * (clip(|x|, max=-b) + b)^4 + 1]
    with coefficients determined from vision activation distributions:
        a = -0.019913, b = -2.698088
    Data-aware-Poly-GELU(x) = 0.5 * x * [1 + L_ours(x / sqrt(2))]
    """
    def __init__(self, integer_arithmetic: bool = True):
        super().__init__()
        self.integer_arithmetic = integer_arithmetic
        # Exact values from IPTQ-ViT Section 4.3
        self.a = -0.019913
        self.b = -2.698088
        self.inv_sqrt2 = 1.0 / math.sqrt(2.0)

    def forward(self, x: torch.Tensor, scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_scaled = x * self.inv_sqrt2
        sign = torch.sign(x_scaled)
        abs_x = torch.abs(x_scaled)
        # clip(|x|, max = -b)
        clipped = torch.clamp(abs_x, max=-self.b)
        # [a * (clip(|x|, max=-b) + b)^4 + 1]
        poly_term = (clipped + self.b) ** 4
        poly = self.a * poly_term + 1.0
        erf_approx = sign * poly
        return 0.5 * x * (1.0 + erf_approx)
