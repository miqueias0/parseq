"""INT-FlashAttention Implementation (arXiv:2409.16997v2)
Chen et al., 'INT-FLASHATTENTION: ENABLING FLASH ATTENTION FOR INT8 QUANTIZATION',
Peking University / Baichuan Inc. / Beihang University, 2024.

Architecture Hierarchy:
- Level 1 (Golden Baseline): Standard FP32 Attention reference.
- Level 2 (PyTorch Mathematical Reference): Pure INT8 tensors (torch.int8),
  exact INT8 x INT8 -> INT32 GEMMs (via hardware torch._int_mm on CUDA Tensor Cores & CPU),
  online softmax and block tiling (Algorithm 1 simulation).
- Level 3 (Performance Kernels & Wrappers): Module interfaces and wrappers for integration
  with Vision Transformers (e.g. PARSeq) and high-performance fused execution.
"""

import math
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


def int8_matmul_int32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Executes general matrix multiplication with INT8 inputs and an INT32 accumulator.
    Strictly verifies that inputs are of type torch.int8 and produces torch.int32.

    On CUDA, dispatches to hardware Tensor Cores via torch._int_mm with automatic padding
    for small M (M <= 16) and N not divisible by 8 as required by cuBLAS / CUTLASS.
    On CPU, dispatches to torch._int_mm or integer arithmetic preserving INT32 precision.

    Args:
        a: Tensor of shape (..., M, K) in torch.int8
        b: Tensor of shape (..., K, N) in torch.int8

    Returns:
        Tensor of shape (..., M, N) in torch.int32
    """
    if a.dtype != torch.int8 or b.dtype != torch.int8:
        raise TypeError(
            f"int8_matmul_int32 requires both inputs to be torch.int8. "
            f"Got a.dtype={a.dtype}, b.dtype={b.dtype}."
        )

    M, K = a.shape[-2], a.shape[-1]
    K2, N = b.shape[-2], b.shape[-1]
    if K != K2:
        raise ValueError(f"Incompatible inner dimensions for matrix multiplication: {K} vs {K2}.")

    orig_batch = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    a_exp = a.expand(*orig_batch, M, K).contiguous().reshape(-1, M, K)
    b_exp = b.expand(*orig_batch, K, N).contiguous().reshape(-1, K, N)
    batch_size = a_exp.shape[0]

    # Tracing / ONNX export fallback
    # TensorRT's ONNX parser does not support INT32 input for MatMul; Float32 MatMul allows
    # ONNX export and enables TensorRT to optimize and fuse into INT8 Tensor Core execution.
    if torch.jit.is_tracing() or (hasattr(torch.onnx, "is_in_onnx_export") and torch.onnx.is_in_onnx_export()):
        out = torch.matmul(a_exp.to(torch.float32), b_exp.to(torch.float32))
        return out.reshape(*orig_batch, M, N)

    # CUDA Tensor Core dimension alignment
    pad_m = (32 - M) if (a.is_cuda and M <= 16) else 0
    pad_n = (8 - (N % 8)) if (a.is_cuda and N % 8 != 0) else 0
    pad_k = (8 - (K % 8)) if (a.is_cuda and K % 8 != 0) else 0

    try:
        if pad_m > 0 or pad_n > 0 or pad_k > 0:
            a_padded = F.pad(a_exp, (0, pad_k, 0, pad_m))
            b_padded = F.pad(b_exp, (0, pad_n, 0, pad_k))
            out_padded = torch.empty((batch_size, M + pad_m, N + pad_n), dtype=torch.int32, device=a.device)
            for i in range(batch_size):
                out_padded[i] = torch._int_mm(a_padded[i], b_padded[i])
            out = out_padded[:, :M, :N]
        else:
            out = torch.empty((batch_size, M, N), dtype=torch.int32, device=a.device)
            for i in range(batch_size):
                out[i] = torch._int_mm(a_exp[i], b_exp[i])
    except (RuntimeError, NotImplementedError):
        # Fallback for platforms where torch._int_mm CUDA kernel is not compiled (e.g. Windows PyTorch builds)
        # Note: torch.matmul on float32 representation of int8 values has 24-bit mantissa precision,
        # which is 100% bit-exact for int8 x int8 inner products with K <= 512.
        out = torch.matmul(a_exp.to(torch.float32), b_exp.to(torch.float32)).to(torch.int32)

    return out.reshape(*orig_batch, M, N)


def exact_sdpa_fp32(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Level 1 Baseline: Exact scaled dot-product attention in FP32."""
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    return F.scaled_dot_product_attention(q.float(), k.float(), v.float(), attn_mask=attn_mask, scale=scale)


def int_flashattention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    attn_mask: Optional[torch.Tensor] = None,
    block_r: int = 64,
    block_c: int = 64,
    bits: int = 8,
    v_quant_mode: str = "per_tensor",
    return_diagnostics: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
    """Level 2 Reference: PyTorch Mathematical Reference Implementation of INT-FlashAttention
    strictly adhering to Algorithm 1 of arXiv:2409.16997v2 (Chen et al., 2024).

    NOTE ON SCIENTIFIC VALIDATION:
    This function is a high-fidelity mathematical reference in PyTorch that simulates the SRAM
    tiling algorithm. It uses genuine INT8 inputs (torch.int8) and hardware INT32 accumulators
    (int8_matmul_int32). It serves as the golden mathematical reference for validation,
    ablation studies, and error decomposition. High-performance speedup requires low-level
    fused SRAM kernels (e.g. Triton/CUDA).

    Algorithm 1 Workflow:
    1. Linear symmetric token-level quantization of Q and K into INT8 with scales SQ, SK.
    2. Quantization of V into INT8 with scale SV (per-tensor or per-head).
    3. Block-tiled computation (Br x Bc) emulating SRAM-resident attention:
       - GEMM-1: S_i^(j) = scale * diag(SQ_i) * (Q_int_i @ K_int_j^T) * diag(SK_j)
         where (Q_int_i @ K_int_j^T) is strictly INT8 x INT8 -> INT32.
       - Online Softmax: tracks running rowmax m_i and exponential sum l_i.
       - Attention Matrix: P_i^(j) = round(R * exp(S_i^(j) - m_new)) in INT8 [0, R].
       - GEMM-2: P_i^(j) @ V_int_j is strictly INT8 x INT8 -> INT32.
       - Online Output Update: O_i^(j) = exp(m_old - m_new) * O_i^(j-1) + (P_i^(j) @ V_int_j)_INT32.
    4. Normalization and Dequantization: O_i = diag(l_i)^(-1) * O_i * SV.

    Args:
        q: Query tensor of shape (B, num_heads, N, head_dim)
        k: Key tensor of shape (B, num_heads, N, head_dim)
        v: Value tensor of shape (B, num_heads, N, head_dim)
        scale: Scaling factor (default: 1.0 / sqrt(head_dim) for Transformer scaled attention;
               pass 1.0 for unscaled dot-product attention as written in Algorithm 1)
        attn_mask: Optional attention mask
        block_r: Row block size Br (default: 64)
        block_c: Column block size Bc (default: 64)
        bits: Bit-width for integer quantization (default: 8)
        v_quant_mode: 'per_tensor' (scalar SV for entire V, as in paper) or
                      'per_head' (SV per batch and head, recommended for multi-head ViT)
        return_diagnostics: If True, returns (output, diagnostics_dict)

    Returns:
        Output tensor O of shape (B, num_heads, N, head_dim), or (O, diagnostics)
    """
    B, H, N, d = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(d)

    R = float((1 << (bits - 1)) - 1)  # 127 for INT8 (I8 in [-128, 127])
    device = q.device
    input_dtype = q.dtype

    # Step 1: Token-level symmetric quantization for Q and K (Section 3.2, p. 4)
    # SQ = rowmax(|Q|) / R, SK = rowmax(|K|) / R
    q_max = q.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    k_max = k.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    s_q = (q_max / R).to(torch.float32)  # (B, H, N, 1)
    s_k = (k_max / R).to(torch.float32)  # (B, H, N, 1)

    # Cast strictly to torch.int8 in [-128, 127]
    q_int = torch.clamp(torch.round(q.to(torch.float32) / s_q), -R - 1, R).to(torch.int8)
    k_int = torch.clamp(torch.round(k.to(torch.float32) / s_k), -R - 1, R).to(torch.int8)

    # Step 2: Quantization for V (Section 3.2, p. 4)
    if v_quant_mode == "per_head":
        # Multi-head adaptation: SV per batch and head (B, H, 1, 1)
        v_max = v.abs().amax(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    elif v_quant_mode == "per_tensor":
        # Strict paper specification: single scalar SV for the entire batched tensor
        v_max = v.abs().amax().clamp(min=1e-8)
    else:
        raise ValueError(f"Unknown v_quant_mode: '{v_quant_mode}'. Choose 'per_tensor' or 'per_head'.")

    s_v = (v_max / R).to(torch.float32)
    v_int = torch.clamp(torch.round(v.to(torch.float32) / s_v), -R - 1, R).to(torch.int8)

    # Validate INT8 datatypes
    assert q_int.dtype == torch.int8, f"q_int must be torch.int8, got {q_int.dtype}"
    assert k_int.dtype == torch.int8, f"k_int must be torch.int8, got {k_int.dtype}"
    assert v_int.dtype == torch.int8, f"v_int must be torch.int8, got {v_int.dtype}"

    # Diagnostics dictionary for scientific audits
    diagnostics: Dict[str, Any] = {}
    if return_diagnostics:
        diagnostics["q_int_dtype"] = str(q_int.dtype)
        diagnostics["k_int_dtype"] = str(k_int.dtype)
        diagnostics["v_int_dtype"] = str(v_int.dtype)
        diagnostics["q_int_range"] = (int(q_int.min()), int(q_int.max()))
        diagnostics["k_int_range"] = (int(k_int.min()), int(k_int.max()))
        diagnostics["v_int_range"] = (int(v_int.min()), int(v_int.max()))
        diagnostics["gemm1_dtypes"] = []
        diagnostics["gemm2_dtypes"] = []
        diagnostics["p_int_dtypes"] = []
        diagnostics["p_int_ranges"] = []

    O_blocks = []
    Tr = (N + block_r - 1) // block_r
    Tc = (N + block_c - 1) // block_c

    # Outer loop: iterate over row blocks of Q (Line 4-5 of Algorithm 1)
    for i in range(Tr):
        r_start = i * block_r
        r_end = min((i + 1) * block_r, N)
        actual_br = r_end - r_start

        q_i = q_int[:, :, r_start:r_end, :]  # (B, H, actual_br, d) in torch.int8
        s_qi = s_q[:, :, r_start:r_end, :]   # (B, H, actual_br, 1) in float32

        # SRAM on-chip accumulator buffers initialized in FP32
        O_i = torch.zeros((B, H, actual_br, d), dtype=torch.float32, device=device)
        l_i = torch.zeros((B, H, actual_br, 1), dtype=torch.float32, device=device)
        m_i = torch.full((B, H, actual_br, 1), -float("inf"), dtype=torch.float32, device=device)

        # Inner loop: iterate over column blocks of K and V (Line 7-8 of Algorithm 1)
        for j in range(Tc):
            c_start = j * block_c
            c_end = min((j + 1) * block_c, N)

            k_j = k_int[:, :, c_start:c_end, :]  # (B, H, actual_bc, d) in torch.int8
            v_j = v_int[:, :, c_start:c_end, :]  # (B, H, actual_bc, d) in torch.int8
            s_kj = s_k[:, :, c_start:c_end, :]   # (B, H, actual_bc, 1) in float32

            # Line 9: S_i^(j) = scale * diag(SQ_i) * (Q_i @ K_j^T) * diag(SK_j)
            # GEMM-1: INT8 x INT8 -> INT32
            gemm1 = int8_matmul_int32(q_i, k_j.transpose(-2, -1).contiguous())
            if not torch.jit.is_tracing():
                assert gemm1.dtype == torch.int32, f"gemm1 must be torch.int32, got {gemm1.dtype}"

            # Dequantize S_ij to float32 using scaling factors and attention scale
            S_ij = (gemm1.to(torch.float32) * s_qi * s_kj.transpose(-2, -1)) * scale

            if attn_mask is not None:
                mask_slice = attn_mask[..., r_start:r_end, c_start:c_end]
                S_ij = S_ij + mask_slice

            # Line 10: m_i^(j) = max(m_i^(j-1), rowmax(S_i^(j)))
            m_ij = S_ij.amax(dim=-1, keepdim=True)
            m_new = torch.maximum(m_i, m_ij)

            # Line 11: P_i^(j) = round(R * exp(S_i^(j) - m_new)) in INT8 [0, R]
            exp_s = torch.exp(S_ij - m_new)
            P_ij = torch.clamp(torch.round(R * exp_s), 0.0, R).to(torch.int8)
            assert P_ij.dtype == torch.int8, f"P_ij must be torch.int8, got {P_ij.dtype}"

            # Line 12: l_i^(j) = exp(m_old - m_new) * l_i^(j-1) + rowsum(P_i^(j))
            alpha = torch.exp(m_i - m_new)
            l_i = alpha * l_i + P_ij.to(torch.float32).sum(dim=-1, keepdim=True)

            # Line 13: O_i^(j) = diag(alpha) * O_i^(j-1) + P_ij @ V_j
            # GEMM-2: INT8 x INT8 -> INT32
            gemm2 = int8_matmul_int32(P_ij, v_j.contiguous())
            if not torch.jit.is_tracing():
                assert gemm2.dtype == torch.int32, f"gemm2 must be torch.int32, got {gemm2.dtype}"

            # Accumulate in FP32 on chip
            O_i = alpha * O_i + gemm2.to(torch.float32)

            # Update running row maximum
            m_i = m_new

            if return_diagnostics and i == 0 and j == 0:
                diagnostics["gemm1_dtypes"].append(str(gemm1.dtype))
                diagnostics["gemm2_dtypes"].append(str(gemm2.dtype))
                diagnostics["p_int_dtypes"].append(str(P_ij.dtype))
                diagnostics["p_int_ranges"].append((int(P_ij.min()), int(P_ij.max())))

        # Line 16: O_i = diag(l_i)^(-1) * O_i * SV (Dequantization to float)
        norm_factor = (l_i.clamp(min=1e-8)) ** -1
        O_i = (O_i * norm_factor) * s_v
        O_blocks.append(O_i)

    # Line 17 & 19: Concatenate row blocks to form output O
    if len(O_blocks) == 1:
        O = O_blocks[0]
    else:
        O = torch.cat(O_blocks, dim=2)

    output = O.to(dtype=input_dtype)
    if return_diagnostics:
        return output, diagnostics
    return output


class INTFlashAttention(nn.Module):
    """Level 3 Module: INT-FlashAttention Module for Transformer Attention blocks.
    Executes true INT8 quantization for Q, K, V and P with INT32 GEMM accumulation.

    Reference:
        Chen et al., 'INT-FLASHATTENTION: ENABLING FLASH ATTENTION FOR INT8 QUANTIZATION',
        arXiv:2409.16997v2, 2024.
    """
    def __init__(
        self,
        embed_dim: int = 384,
        num_heads: int = 6,
        block_r: int = 64,
        block_c: int = 64,
        bits: int = 8,
        v_quant_mode: str = "per_tensor",
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
        self.v_quant_mode = v_quant_mode
        self.use_plugin = use_plugin

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """Executes INT-FlashAttention Algorithm 1.
        q, k, v are tensors of shape (B, num_heads, N, head_dim).
        """
        if self.use_plugin:
            from strhub.quant.plugins.trt_plugins import INTFlashAttentionPluginOp
            return INTFlashAttentionPluginOp.apply(q, k, v, self.scale)

        return int_flashattention_forward(
            q=q,
            k=k,
            v=v,
            scale=self.scale,
            attn_mask=attn_mask,
            block_r=self.block_r,
            block_c=self.block_c,
            bits=self.bits,
            v_quant_mode=self.v_quant_mode,
            return_diagnostics=return_diagnostics,
        )


class FusedINTFlashAttentionWrapper(nn.Module):
    """Wraps timm Attention to use INT-FlashAttention for the Q@K -> Softmax -> Attn@V block.
    Integrates seamlessly into PARSeq ViT Encoder blocks.
    """
    def __init__(
        self,
        original_attn: nn.Module,
        block_r: int = 64,
        block_c: int = 64,
        bits: int = 8,
        v_quant_mode: str = "per_tensor",
    ):
        super().__init__()
        self.attn = original_attn
        embed_dim = (
            self.attn.attn_dim
            if hasattr(self.attn, "attn_dim")
            else (self.attn.qkv.out_features // 3 if hasattr(self.attn, "qkv") else 384)
        )
        self.int_flash_attn = INTFlashAttention(
            embed_dim=embed_dim,
            num_heads=self.attn.num_heads,
            block_r=block_r,
            block_c=block_c,
            bits=bits,
            v_quant_mode=v_quant_mode,
        )

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if "attn" in self.__dict__:
                return getattr(self.attn, name)
            raise

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        B, N, C = x.shape
        # QKV linear projection
        qkv = self.attn.qkv(x).reshape(B, N, 3, self.attn.num_heads, self.attn.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Handle q_norm / k_norm robustly across timm versions
        if hasattr(self.attn, "q_norm") and self.attn.q_norm is not None:
            q = self.attn.q_norm(q)
        if hasattr(self.attn, "k_norm") and self.attn.k_norm is not None:
            k = self.attn.k_norm(k)

        # Fused INT8 FlashAttention forward pass (Algorithm 1)
        x = self.int_flash_attn(q, k, v, attn_mask=attn_mask)

        x = x.transpose(1, 2).reshape(B, N, getattr(self.attn, "attn_dim", C))
        if hasattr(self.attn, "norm") and self.attn.norm is not None:
            x = self.attn.norm(x)
        x = self.attn.proj(x)
        x = self.attn.proj_drop(x)
        return x
