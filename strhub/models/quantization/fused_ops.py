# Scene Text Recognition Model Hub - Jetfire Fused Non-Linear Operators
# Reference: Xi et al., "Jetfire: Efficient and Accurate Transformer Pretraining with INT8 Data Flow
# and Per-Block Quantization", ICML 2024 (arXiv:2403.12422, Section 6)

import math
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ---------------------------------------------------------------------------
# Triton Kernels for CUDA (sm_75 / Turing RTX 2080 Ti and modern GPUs)
# ---------------------------------------------------------------------------
if HAS_TRITON:
    @triton.jit
    def _triton_fused_int8_gelu_kernel(
        x_ptr,
        scale_x_ptr,
        y_ptr,
        scale_y_ptr,
        total_elements,
        block_size: tl.constexpr,
    ):
        """Triton Kernel: Fused INT8 -> FP32 Dequant -> GELU -> Per-Block Requant to INT8."""
        pid = tl.program_id(0)
        block_start = pid * block_size
        offsets = block_start + tl.arange(0, block_size)
        mask = offsets < total_elements

        # 1. Load INT8 input and block scale
        x_int8 = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.float32)
        s_x = tl.load(scale_x_ptr + pid)

        # 2. Dequantize to FP32 in registers/SRAM
        x_fp = x_int8 * s_x

        # 3. Compute GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
        # Using exact high-speed tanh polynomial approximation in registers
        # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        sqrt_2_over_pi = 0.7978845608028654
        tanh_arg = sqrt_2_over_pi * (x_fp + 0.044715 * x_fp * x_fp * x_fp)
        # Numerical clamp for tanh stability
        tanh_arg_clamped = tl.clamp(tanh_arg, -10.0, 10.0)
        # exp(2*z) formulation for tanh
        exp_val = tl.exp(2.0 * tanh_arg_clamped)
        tanh_val = (exp_val - 1.0) / (exp_val + 1.0)
        y_fp = 0.5 * x_fp * (1.0 + tanh_val)

        # 4. Compute block maximum for INT8 requantization
        y_abs = tl.abs(y_fp)
        block_max = tl.max(tl.where(mask, y_abs, 0.0), axis=0)
        s_y = tl.maximum(block_max / 127.0, 1e-8)

        # 5. Quantize back to INT8 in registers
        y_int8 = tl.clamp(tl.math.round(y_fp / s_y), -128.0, 127.0).to(tl.int8)

        # 6. Store INT8 output and scale factor
        tl.store(y_ptr + offsets, y_int8, mask=mask)
        tl.store(scale_y_ptr + pid, s_y)


class JetfireFusedGELU(nn.Module):
    """Fused GELU Operator with INT8 Data Flow (Xi et al., Jetfire ICML 2024 Section 6).
    
    Eliminates QCD memory overhead by fusing:
    [INT8 In, Scale In] -> Load to SRAM -> FP32 Dequant -> GELU -> FP32 Requant -> [INT8 Out, Scale Out].
    """

    def __init__(self, block_size: int = 32):
        super().__init__()
        self.block_size = block_size

    def forward(
        self,
        x: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        return_int8: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with dual CUDA/Triton and CPU vectorized execution."""
        orig_shape = x.shape
        device = x.device

        # If continuous float input is supplied, quantize first to simulate Jetfire flow
        if scale is None:
            # Per-block symmetric INT8 quantization
            x_flat = x.reshape(-1)
            pad = (self.block_size - (x_flat.numel() % self.block_size)) % self.block_size
            if pad > 0:
                x_padded = F.pad(x_flat, (0, pad))
            else:
                x_padded = x_flat
            
            blocks = x_padded.view(-1, self.block_size)
            scale = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
            x_int8 = (blocks / scale).round().clamp(-128, 127).to(torch.int8)
        else:
            x_int8 = x
            pad = 0

        if x_int8.is_cuda and HAS_TRITON and x_int8.dtype == torch.int8:
            num_elements = x_int8.numel()
            grid = (triton.cdiv(num_elements, self.block_size),)
            y_int8 = torch.empty_like(x_int8)
            scale_y = torch.empty((grid[0], 1), dtype=torch.float32, device=device)

            _triton_fused_int8_gelu_kernel[grid](
                x_int8.contiguous(),
                scale.contiguous(),
                y_int8,
                scale_y,
                num_elements,
                block_size=self.block_size,
            )
            if return_int8:
                return y_int8, scale_y
            y_deq = y_int8.float() * scale_y
            if pad > 0:
                y_deq = y_deq.view(-1)[:orig_shape.numel()]
            return y_deq.reshape(orig_shape)

        # Vectorized PyTorch Implementation (CPU / universal fallback)
        if not return_int8 and scale is None:
            return F.gelu(x)

        if x_int8.dtype == torch.int8:
            if scale.dim() < x_int8.dim():
                scale_exp = scale.view(*x_int8.shape[:-1], -1)
            else:
                scale_exp = scale
            x_fp = x_int8.float() * scale_exp
        else:
            x_fp = x_int8.float()

        y_fp = F.gelu(x_fp)

        if return_int8:
            y_flat = y_fp.view(-1)
            pad_y = (self.block_size - (y_flat.numel() % self.block_size)) % self.block_size
            if pad_y > 0:
                y_pad = F.pad(y_flat, (0, pad_y))
            else:
                y_pad = y_flat
            y_blocks = y_pad.view(-1, self.block_size)
            scale_y = y_blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
            y_int8 = (y_blocks / scale_y).round().clamp(-128, 127).to(torch.int8)
            return y_int8, scale_y

        if pad > 0:
            y_fp = y_fp.view(-1)[:orig_shape.numel()]
        return y_fp.reshape(orig_shape).to(x.dtype)


class JetfireFusedLayerNorm(nn.Module):
    """Fused LayerNorm with INT8 Data Flow (Xi et al., Jetfire ICML 2024 Section 6).
    
    Computes normalization across channel dimension directly from INT8 tokens,
    applies learnable affine scale and bias, and optionally requantizes directly to INT8.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5, block_size: int = 32):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.block_size = block_size
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(
        self,
        x: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        return_int8: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Dequantize in registers/SRAM
        if scale is not None and x.dtype == torch.int8:
            x_fp = x.float() * scale
        else:
            x_fp = x.float()

        # Compute LayerNorm
        mean = x_fp.mean(dim=-1, keepdim=True)
        var = ((x_fp - mean) ** 2).mean(dim=-1, keepdim=True)
        x_norm = (x_fp - mean) / torch.sqrt(var + self.eps)
        out_fp = x_norm * self.weight + self.bias

        if return_int8:
            # Token/channel wise INT8 requantization
            s_out = out_fp.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
            out_int8 = (out_fp / s_out).round().clamp(-128, 127).to(torch.int8)
            return out_int8, s_out

        return out_fp.to(x.dtype)


class JetfireFusedResidualAdd(nn.Module):
    """Fused Residual Add Operator with INT8 Data Flow.
    
    Loads two INT8 tensors with their respective scaling factors, adds them in SRAM,
    and returns either the combined float tensor or requantized INT8 output.
    """

    def __init__(self, block_size: int = 32):
        super().__init__()
        self.block_size = block_size

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        scale1: Optional[torch.Tensor] = None,
        scale2: Optional[torch.Tensor] = None,
        return_int8: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Dequantize both operands in registers
        x1_fp = (x1.float() * scale1) if (scale1 is not None and x1.dtype == torch.int8) else x1.float()
        x2_fp = (x2.float() * scale2) if (scale2 is not None and x2.dtype == torch.int8) else x2.float()

        # Add in registers
        y_fp = x1_fp + x2_fp

        if return_int8:
            s_y = y_fp.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
            y_int8 = (y_fp / s_y).round().clamp(-128, 127).to(torch.int8)
            return y_int8, s_y

        return y_fp.to(x1.dtype)
