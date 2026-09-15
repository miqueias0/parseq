# Scene Text Recognition Model Hub - INT8 Quantization Core
# Implements Straight-Through Estimator (STE) and Quantization Granularities
# References:
#   - I-BERT (Kim et al., ICML 2021)
#   - Jetfire: INT8 Data Flow and Per-Block Quantization (Xi et al., ICML 2024)

import math
from enum import Enum
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantGranularity(str, Enum):
    PER_TENSOR = "per_tensor"
    PER_CHANNEL = "per_channel"
    PER_TOKEN = "per_token"
    PER_BLOCK = "per_block"


class STEQuantizeFunction(torch.autograd.Function):
    """Straight-Through Estimator (STE) for symmetric uniform quantization.
    
    Forward:
        q = clamp(round(x / scale), qmin, qmax)
        x_hat = q * scale
        
    Backward:
        grad_x = grad_output if (qmin <= x / scale <= qmax) else 0
        grad_scale = sum(grad_output * (q - x / scale)) if scale requires_grad
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: torch.Tensor, qmin: int, qmax: int):
        x_scaled = x / scale
        x_clamped = torch.clamp(torch.round(x_scaled), qmin, qmax)
        ctx.save_for_backward(x_scaled, scale)
        ctx.qmin = qmin
        ctx.qmax = qmax
        return x_clamped * scale

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_scaled, scale = ctx.saved_tensors
        qmin, qmax = ctx.qmin, ctx.qmax
        
        # Straight-Through Estimator: gradient passes through when input is within saturation bounds
        mask = (x_scaled >= qmin) & (x_scaled <= qmax)
        grad_x = grad_output * mask.to(grad_output.dtype)
        
        # Gradient for scale if learning scale
        grad_scale = None
        if scale.requires_grad:
            q = torch.clamp(torch.round(x_scaled), qmin, qmax)
            # Analytical derivative w.r.t scale: d(x_hat)/d(scale) = (q - x_scaled)
            grad_scale_val = grad_output * (q - x_scaled)
            # Reduce across dimensions to match scale's shape
            if scale.dim() == 0 or scale.numel() == 1:
                grad_scale = grad_scale_val.sum()
            else:
                dims_to_sum = [i for i, d in enumerate(scale.shape) if d == 1 and x_scaled.shape[i] != 1]
                grad_scale = grad_scale_val.sum(dim=dims_to_sum, keepdim=True)
                
        return grad_x, grad_scale, None, None


def ste_quantize(x: torch.Tensor, scale: torch.Tensor, qmin: int = -128, qmax: int = 127) -> torch.Tensor:
    """Convenience wrapper for STE quantization."""
    return STEQuantizeFunction.apply(x, scale, qmin, qmax)


class SymmetricUniformQuantizer(nn.Module):
    """Symmetric Uniform INT8 Quantizer with Straight-Through Estimator (STE).
    
    Supports:
        - Per-tensor, per-channel, and per-token scaling.
        - Calibration mode (collecting min/max statistics).
        - QAT mode (fine-tuning with backprop via STE).
        - Freeze mode (inference with fixed scales).
    """

    def __init__(
        self,
        bits: int = 8,
        granularity: Union[QuantGranularity, str] = QuantGranularity.PER_TENSOR,
        ch_axis: int = 0,
        learnable_scale: bool = False,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.bits = bits
        self.granularity = QuantGranularity(granularity)
        self.ch_axis = ch_axis
        self.learnable_scale = learnable_scale
        self.eps = eps
        
        # Buffer for calibrated scale
        self.register_buffer("scale", None)
        self.calibrated = False
        self.training_mode = False

    @property
    def qmin(self) -> int:
        return -(2 ** (self.bits - 1))

    @property
    def qmax(self) -> int:
        return (2 ** (self.bits - 1)) - 1

    def compute_scale(self, x: torch.Tensor) -> torch.Tensor:
        """Computes quantization scale factor S = max(|x|) / (2^(b-1) - 1)."""
        if self.granularity == QuantGranularity.PER_TENSOR:
            max_val = torch.max(torch.abs(x))
            scale = max_val / self.qmax
        elif self.granularity == QuantGranularity.PER_CHANNEL:
            dims = [i for i in range(x.dim()) if i != self.ch_axis]
            if len(dims) > 0:
                max_val = torch.amax(torch.abs(x), dim=dims, keepdim=True)
            else:
                max_val = torch.abs(x)
            scale = max_val / self.qmax
        elif self.granularity == QuantGranularity.PER_TOKEN:
            # Per-token scale along the last dimension (e.g. feature dim C)
            max_val = torch.amax(torch.abs(x), dim=-1, keepdim=True)
            scale = max_val / self.qmax
        else:
            raise ValueError(f"Unsupported granularity: {self.granularity}")
            
        return torch.clamp(scale, min=self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bits >= 16:
            return x

        if self.calibrated and self.scale is not None:
            scale = self.scale
        else:
            scale = self.compute_scale(x)

        if self.training:
            # Quantization-Aware Training (QAT): gradients flow via STE
            return ste_quantize(x, scale, self.qmin, self.qmax)
        else:
            # Fast inference path
            x_scaled = x / scale
            x_clamped = torch.clamp(torch.round(x_scaled), self.qmin, self.qmax)
            return x_clamped * scale

    def extra_repr(self) -> str:
        return f"bits={self.bits}, granularity={self.granularity.value}, calibrated={self.calibrated}"


class PerBlockQuantizer(nn.Module):
    """Jetfire Per-Block Quantizer (Xi et al., ICML 2024).
    
    Partitions 2D activation or weight matrices into blocks of size [block_r, block_c]
    (e.g., 64x64) and applies independent scale factors per block.
    This effectively confines activation channel outliers to their respective blocks,
    preventing error propagation across the entire token sequence.
    """

    def __init__(self, block_r: int = 64, block_c: int = 64, bits: int = 8, eps: float = 1e-8):
        super().__init__()
        self.block_r = block_r
        self.block_c = block_c
        self.bits = bits
        self.eps = eps

    @property
    def qmin(self) -> int:
        return -(2 ** (self.bits - 1))

    @property
    def qmax(self) -> int:
        return (2 ** (self.bits - 1)) - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1])
        M, K = x_2d.shape
        
        pad_m = (self.block_r - (M % self.block_r)) % self.block_r
        pad_k = (self.block_c - (K % self.block_c)) % self.block_c
        
        if pad_m > 0 or pad_k > 0:
            x_padded = F.pad(x_2d, (0, pad_k, 0, pad_m))
        else:
            x_padded = x_2d
            
        M_pad, K_pad = x_padded.shape
        # Reshape to [num_blocks_m, block_r, num_blocks_k, block_c]
        blocks = x_padded.view(
            M_pad // self.block_r, self.block_r,
            K_pad // self.block_c, self.block_c
        ).permute(0, 2, 1, 3) # [num_bm, num_bk, block_r, block_c]
        
        # Scale per block: max over block dimensions
        scale = torch.amax(torch.abs(blocks), dim=(-2, -1), keepdim=True) / self.qmax
        scale = torch.clamp(scale, min=self.eps)
        
        if self.training:
            blocks_q = ste_quantize(blocks, scale, self.qmin, self.qmax)
        else:
            blocks_q = torch.clamp(torch.round(blocks / scale), self.qmin, self.qmax) * scale
            
        out_2d = blocks_q.permute(0, 2, 1, 3).reshape(M_pad, K_pad)
        if pad_m > 0 or pad_k > 0:
            out_2d = out_2d[:M, :K]
            
        return out_2d.reshape(*orig_shape)
