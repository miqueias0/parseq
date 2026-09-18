import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftmaxFP32(nn.Module):
    """Standard FP32 Softmax."""
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(x, dim=self.dim)


class IBERTSoftmax(nn.Module):
    """Integer-Only Softmax from I-BERT (Kim et al., ICML 2021, Section 3.5).
    Decomposes negative values into x = (-ln 2) * z + p with p in (-ln 2, 0].
    Approximates exp(p) via:
        L(p) = 0.3585 * (p + 1.353)^2 + 0.344
    Then:
        i-exp(x) = L(p) >> z
    """
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim
        self.ln2 = math.log(2.0)
        self.a = 0.3585
        self.b = 1.353
        self.c = 0.344

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Subtract max for numerical stability
        max_val = torch.max(x, dim=self.dim, keepdim=True)[0]
        x_shifted = x - max_val  # All values <= 0

        # Decomposition: x_shifted = (-ln 2)*z + p
        # z = floor(-x_shifted / ln2)
        z = torch.floor(-x_shifted / self.ln2)
        p = x_shifted + z * self.ln2  # p in (-ln 2, 0]

        # Polynomial evaluation for exp(p)
        poly = self.a * (p + self.b) ** 2 + self.c

        # Bit shifting representation: 2^(-z) * L(p)
        exp_approx = poly * torch.pow(2.0, -z)

        # Normalize
        sum_exp = torch.sum(exp_approx, dim=self.dim, keepdim=True)
        return exp_approx / torch.clamp(sum_exp, min=1e-8)


class IViTShiftmax(nn.Module):
    """Shiftmax from I-ViT (Li et al., 2022).
    Uses base-2 reformulation and power-of-two bit shifting:
        e^(S * Q) = 2^(S * Q * log2(e))
    Approximates fractional component with [S * (-r)] / 2 + 1.
    """
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim
        self.log2_e = math.log2(math.e)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_val = torch.max(x, dim=self.dim, keepdim=True)[0]
        x_shifted = (x - max_val) * self.log2_e  # base-2 scaling

        # Decompose into integer q and fractional r
        q = torch.floor(-x_shifted)
        r = -x_shifted - q  # fractional in [0, 1)

        # Linear approximation: 2^(-r) approx 1 - 0.5 * r
        frac_approx = torch.clamp(1.0 - 0.5 * r, min=0.0)
        exp_approx = frac_approx * torch.pow(2.0, -q)

        sum_exp = torch.sum(exp_approx, dim=self.dim, keepdim=True)
        return exp_approx / torch.clamp(sum_exp, min=1e-8)


class IPTQBitSoftmax(nn.Module):
    """Efficient Bit-Softmax from IPTQ-ViT (CVPR 2024, Section 4.4).
    Uses first-order Taylor expansion with bit shifts:
        2^X = e^(ln 2 * X) approx 1 + ln 2 * X
    Approximates ln 2 as binary (0.1011)_b:
        Phi(x) = (x >> 1) + (x >> 3) + (x >> 4)
    Computes integer division IntDiv with scaling M.
    """
    def __init__(self, dim: int = -1, bits: int = 8, M: int = 24):
        super().__init__()
        self.dim = dim
        self.bits = bits
        self.M = M
        self.log2_e = math.log2(math.e)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_val = torch.max(x, dim=self.dim, keepdim=True)[0]
        x_shifted = (x - max_val) * self.log2_e

        q = torch.floor(-x_shifted)
        r = -x_shifted - q  # r in [0, 1)

        # Efficient-Bit-exp approximation of 2^(-r):
        # Phi(r) approx (r * 0.5) + (r * 0.125) + (r * 0.0625) = r * 0.6875 (close to ln 2 = 0.6931)
        phi_r = 0.5 * r + 0.125 * r + 0.0625 * r
        exp_frac = torch.clamp(1.0 - phi_r, min=0.0)

        exp_approx = exp_frac * torch.pow(2.0, -q)

        # Normalization with simulated integer division (IntDiv, Eq. 15)
        sum_exp = torch.sum(exp_approx, dim=self.dim, keepdim=True) + 1e-8
        probs = exp_approx / sum_exp
        return probs
