"""SageAttention Implementation (arXiv:2410.02367v9 - ICLR 2025)
Zhang et al., 'SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration',
Tsinghua University, 2025.

Core Innovations:
1. Smooth Matrix K (Equation 6):
   Subtracts the token-averaged key vector: K_smooth = K - mean(K, dim=-2).
   Softmax translation invariance guarantees:
     softmax(Q (K - mean(K))^T / sqrt(d)) == softmax(Q K^T / sqrt(d))
   This suppresses channel-wise key outliers without any loss of attention accuracy.
2. 8-Bit Quantization & INT8 Tensor Core GEMMs:
   - Q and K_smooth quantized dynamically to INT8 with per-token or per-block scaling.
   - GEMM-1: (Q_int @ K_int^T) executed strictly as INT8 x INT8 -> INT32.
   - Online Softmax: tracks running row-max m and row-sum l.
3. Two Operational Modes:
   - SAGEAttn-B (Algorithm 1): P_tilde in float/half multiplied with V (FP16/FP32).
   - SAGEAttn-vB (fully INT8): P_tilde quantized to INT8 (scale 1/127) and V quantized
     per-channel to INT8, executing GEMM-2 as INT8 x INT8 -> INT32.
"""

import math
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from strhub.quant.int_flashattention import int8_matmul_int32


def smooth_k(k: torch.Tensor) -> torch.Tensor:
    """Applies SageAttention Key Smoothing (Eq. 6 of arXiv:2410.02367v9).
    Subtracts the average key vector across the token dimension (dim=-2):
        gamma(K) = K - mean(K, dim=-2, keepdim=True)

    Proof of Softmax Invariance:
        softmax(q (K - mean(K))^T) = softmax(q K^T - q * mean(K)^T)
    Since q * mean(K)^T is identical for all key positions in a row,
    softmax(z - c) = softmax(z), preserving mathematical equivalence while
    eradicating channel-wise outliers that hinder INT8 quantization.
    """
    mean_k = k.mean(dim=-2, keepdim=True)
    return k - mean_k


def sage_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    attn_mask: Optional[torch.Tensor] = None,
    block_r: int = 64,
    block_c: int = 64,
    bits: int = 8,
    mode: str = "sageattn_b",
    smooth: bool = True,
    return_diagnostics: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
    """Mathematical Reference Implementation of SageAttention (Algorithm 1 from arXiv:2410.02367v9).

    Args:
        q: Query tensor of shape (B, num_heads, N, head_dim)
        k: Key tensor of shape (B, num_heads, S, head_dim)
        v: Value tensor of shape (B, num_heads, S, head_dim)
        scale: Scaling factor (default: 1.0 / sqrt(head_dim))
        attn_mask: Optional attention mask of broadcastable shape
        block_r: Query block size bq (default: 64)
        block_c: Key/Value block size bkv (default: 64)
        bits: Integer bitwidth (default: 8)
        mode: 'sageattn_b' (Algorithm 1 with V in float/half) or
              'sageattn_vb' (both GEMMs in INT8 with per-channel V quant)
        smooth: Whether to apply key smoothing K - mean(K) (default: True)
        return_diagnostics: If True, returns (out, diagnostics_dict)

    Returns:
        Output tensor O of shape (B, num_heads, N, head_dim), or (O, diagnostics)
    """
    B, H, N, d = q.shape
    S = k.shape[-2]
    if scale is None:
        scale = 1.0 / math.sqrt(d)

    R = float((1 << (bits - 1)) - 1)  # 127 for INT8
    device = q.device
    orig_dtype = q.dtype

    # Tracing / ONNX export fallback
    if torch.jit.is_tracing() or (hasattr(torch.onnx, "is_in_onnx_export") and torch.onnx.is_in_onnx_export()):
        k_proc = smooth_k(k) if smooth else k
        q_s = q * scale
        attn = torch.matmul(q_s, k_proc.transpose(-2, -1))
        if attn_mask is not None:
            attn = attn + attn_mask
        p = F.softmax(attn, dim=-1)
        out = torch.matmul(p, v)
        return out.to(orig_dtype)

    # 1. Key Smoothing (Section 4.2, Eq. 6)
    k_smooth = smooth_k(k) if smooth else k

    # 2. Dynamic per-token INT8 quantization for Q and K (Section 4.3)
    # Q is scaled by 1/sqrt(d) prior to quantization as in Algorithm 1 Line 2: (delta_Q, Q_hat) = psi_Q(Q / sqrt(d))
    q_scaled = (q.float() * scale)
    q_max = q_scaled.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    delta_q = q_max / R
    q_int = torch.clamp(torch.round(q_scaled / delta_q), -R - 1, R).to(torch.int8)

    k_max = k_smooth.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    delta_k = k_max / R
    k_int = torch.clamp(torch.round(k_smooth.float() / delta_k), -R - 1, R).to(torch.int8)

    # 3. Optional per-channel quantization for V in SAGEAttn-vB mode (Section 4.3, 4.4)
    if mode == "sageattn_vb":
        # Per-channel quantization across sequence dimension (dim=-2)
        v_max = v.abs().amax(dim=-2, keepdim=True).clamp(min=1e-8)
        delta_v = v_max / R
        v_int = torch.clamp(torch.round(v.float() / delta_v), -R - 1, R).to(torch.int8)
    else:
        v_int = None
        delta_v = None

    # Diagnostics
    diagnostics: Dict[str, Any] = {}
    if return_diagnostics:
        diagnostics["q_int_dtype"] = str(q_int.dtype)
        diagnostics["k_int_dtype"] = str(k_int.dtype)
        diagnostics["smooth_applied"] = smooth
        diagnostics["mode"] = mode
        diagnostics["q_int_range"] = (int(q_int.min()), int(q_int.max()))
        diagnostics["k_int_range"] = (int(k_int.min()), int(k_int.max()))
        if v_int is not None:
            diagnostics["v_int_dtype"] = str(v_int.dtype)
            diagnostics["v_int_range"] = (int(v_int.min()), int(v_int.max()))

    # 4. Tiled Execution over Blocks (Algorithm 1)
    Tm = (N + block_r - 1) // block_r
    Tn = (S + block_c - 1) // block_c
    O_blocks = []

    for i in range(Tm):
        r_start = i * block_r
        r_end = min((i + 1) * block_r, N)
        actual_br = r_end - r_start

        q_i = q_int[:, :, r_start:r_end, :]       # (B, H, actual_br, d) in torch.int8
        delta_qi = delta_q[:, :, r_start:r_end, :] # (B, H, actual_br, 1) in float32

        # SRAM accumulators
        O_i = torch.zeros((B, H, actual_br, d), dtype=torch.float32, device=device)
        l_i = torch.zeros((B, H, actual_br, 1), dtype=torch.float32, device=device)
        m_i = torch.full((B, H, actual_br, 1), -float("inf"), dtype=torch.float32, device=device)

        for j in range(Tn):
            c_start = j * block_c
            c_end = min((j + 1) * block_c, S)
            actual_bc = c_end - c_start

            k_j = k_int[:, :, c_start:c_end, :]       # (B, H, actual_bc, d) in torch.int8
            delta_kj = delta_k[:, :, c_start:c_end, :] # (B, H, actual_bc, 1) in float32

            # GEMM-1: Matmul(Q_hat_i, K_hat_j^T) -> INT32 accumulator (Algorithm 1 Line 9)
            k_j_t = k_j.transpose(-2, -1).contiguous()
            S_int = int8_matmul_int32(q_i, k_j_t)  # (B, H, actual_br, actual_bc) in torch.int32

            # Rescale to float: S_ij = S_int * delta_Q[i] * delta_K[j]
            S_ij = S_int.float() * delta_qi * delta_kj.transpose(-2, -1)

            if attn_mask is not None:
                mask_chunk = attn_mask[..., r_start:r_end, c_start:c_end]
                S_ij = S_ij + mask_chunk

            # Online Softmax update (Algorithm 1 Line 10)
            m_prev = m_i
            row_max_s = S_ij.amax(dim=-1, keepdim=True)
            m_i = torch.maximum(m_prev, row_max_s)

            alpha = torch.exp(m_prev - m_i)
            # Guard against exp(-inf - (-inf)) -> NaN in initial iteration
            alpha = torch.nan_to_num(alpha, nan=0.0)

            P_tilde = torch.exp(S_ij - m_i)  # (B, H, actual_br, actual_bc) in float32
            l_i = alpha * l_i + P_tilde.sum(dim=-1, keepdim=True)

            # GEMM-2: Output accumulation (Algorithm 1 Line 11)
            if mode == "sageattn_vb":
                # SAGEAttn-vB: Fully INT8 second GEMM
                # Quantize P_tilde to INT8: P_int in [0, 127], delta_P = 1/127
                p_int = torch.clamp(torch.round(P_tilde * R), 0.0, R).to(torch.int8)
                v_j_int = v_int[:, :, c_start:c_end, :]  # (B, H, actual_bc, d) in torch.int8
                delta_v_j = delta_v[:, :, :, :]         # per-channel scale (B, H, 1, d)

                # INT8 x INT8 -> INT32
                pv_int = int8_matmul_int32(p_int, v_j_int)
                pv_scaled = pv_int.float() * (1.0 / R) * delta_v_j
                O_i = alpha * O_i + pv_scaled
            else:
                # SAGEAttn-B: P_tilde.to(FP16) @ V_j in FP16/FP32
                v_j = v[:, :, c_start:c_end, :].to(torch.float32)
                O_i = alpha * O_i + (P_tilde @ v_j)

        # Normalization by l_i (Algorithm 1 Line 12)
        O_i = O_i / l_i.clamp(min=1e-8)
        O_blocks.append(O_i)

    O = torch.cat(O_blocks, dim=-2).to(orig_dtype)

    if return_diagnostics:
        return O, diagnostics
    return O


class SageAttention(nn.Module):
    """SageAttention Module (ICLR 2025, arXiv:2410.02367v9)
    Can be used as a drop-in replacement for MultiheadAttention or self/cross attention in Transformers.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        block_r: int = 64,
        block_c: int = 64,
        bits: int = 8,
        mode: str = "sageattn_b",
        smooth: bool = True,
        use_plugin: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.block_r = block_r
        self.block_c = block_c
        self.bits = bits
        self.mode = mode
        self.smooth = smooth
        self.use_plugin = use_plugin

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args:
            q: (B, H, N, D)
            k: (B, H, S, D)
            v: (B, H, S, D)
        """
        if self.use_plugin:
            from strhub.quant.plugins.trt_plugins import SageAttentionPluginOp
            mode_int = 0 if self.mode == "sageattn_b" else 1
            return SageAttentionPluginOp.apply(q, k, v, self.scale, mode_int)

        return sage_attention_forward(
            q=q,
            k=k,
            v=v,
            scale=self.scale,
            attn_mask=attn_mask,
            block_r=self.block_r,
            block_c=self.block_c,
            bits=self.bits,
            mode=self.mode,
            smooth=self.smooth,
        )
