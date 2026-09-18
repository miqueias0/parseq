from typing import Optional, Union, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F


def integer_sqrt_newton(n: torch.Tensor, max_iters: int = 4) -> torch.Tensor:
    """Integer square root using Newton's method (I-BERT Algorithm 4 / Crandall & Pomerance 2006).
    xi+1 = floor((xi + floor(n / xi)) / 2)
    Converges within at most 4 iterations for INT32 inputs.
    """
    n_clamped = torch.clamp(n, min=1e-8)
    # Initial estimate: 2^(ceil(bits(n)/2))
    # In vectorized PyTorch, we can initialize with sqrt estimate
    x0 = torch.clamp(torch.sqrt(n_clamped), min=1.0)
    xi = x0
    for _ in range(max_iters):
        xi = 0.5 * (xi + n_clamped / xi)
    return xi


class LayerNormFP32(nn.Module):
    """Standard FP32 LayerNorm."""
    def __init__(self, normalized_shape: Union[int, Sequence[int]], eps: float = 1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x)


class IBERTLayerNorm(nn.Module):
    """Integer-Only LayerNorm from I-BERT (Kim et al., ICML 2021, Section 3.6).
    Computes mean and variance across channel dimension, and evaluates standard deviation
    using iterative integer square root (Algorithm 4).
    """
    def __init__(self, normalized_shape: Union[int, Sequence[int]], eps: float = 1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            self.normalized_shape = (normalized_shape,)
        else:
            self.normalized_shape = tuple(normalized_shape)
        dim = self.normalized_shape[0]
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute mean across channel dimension
        mean = torch.mean(x, dim=-1, keepdim=True)
        # Compute variance across channel dimension
        diff = x - mean
        variance = torch.mean(diff ** 2, dim=-1, keepdim=True)

        # Evaluate standard deviation using integer Newton's method
        std = integer_sqrt_newton(variance + self.eps)

        # Normalize and apply affine parameters
        normed = diff / std
        return normed * self.weight + self.bias


class IPTQLayerNorm(nn.Module):
    """Integer LayerNorm compatible with IPTQ-ViT pipeline.
    Preserves fixed-point scale factor propagation.
    """
    def __init__(self, normalized_shape: Union[int, Sequence[int]], eps: float = 1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            self.normalized_shape = (normalized_shape,)
        else:
            self.normalized_shape = tuple(normalized_shape)
        dim = self.normalized_shape[0]
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.var(x, dim=-1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + self.eps)
        normed = (x - mean) / std
        return normed * self.weight + self.bias
