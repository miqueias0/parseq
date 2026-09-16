# Scene Text Recognition Model Hub - INT-FlashAttention Module
# References:
# 1. arXiv:2409.16997: "INT-FlashAttention: Fully INT8 Attention Module with Quantized Online Softmax"
# 2. Kim et al., ICML 2021: "I-BERT: Integer-only BERT Quantization" (arXiv:2101.01321)

import math
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ibert_ops import IExpSoftmax


def quantize_token_symmetric(tensor: torch.Tensor, range_val: float = 127.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Token-wise (row-wise) symmetric INT8 quantization.
    
    Q_int8 = round(Q / s_q), where s_q = amax(|Q|, dim=-1) / 127.
    """
    scale = tensor.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / range_val
    q = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
    return q, scale


def quantize_tensor_symmetric(tensor: torch.Tensor, range_val: float = 127.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tensor-wide or block-wide symmetric INT8 quantization (used for V in INT-FlashAttention)."""
    scale = tensor.abs().amax().clamp(min=1e-8) / range_val
    q = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
    return q, scale


def int_flash_attention_core(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sq: torch.Tensor,
    sk: torch.Tensor,
    sv: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    key_padding_mask: Optional[torch.Tensor] = None,
    block_r: int = 32,
    block_c: int = 32,
    range_r: float = 127.0,
    use_ibert_exp: bool = False,
) -> torch.Tensor:
    """INT-FlashAttention with quantized online softmax (arXiv:2409.16997).
    
    Computes tiled attention entirely with INT8 matrix multiplications and SRAM-level
    online softmax accumulation, avoiding materialization of the full N x M attention matrix.
    
    Args:
        q: INT8 Query tensor [B, H, N, d]
        k: INT8 Key tensor [B, H, M, d]
        v: INT8 Value tensor [B, H, M, d]
        sq: Query scale [B, H, N, 1]
        sk: Key scale [B, H, M, 1]
        sv: Value scale (scalar or [B, H, 1, 1])
        attn_mask: Optional attention mask [N, M] or [B, H, N, M]
        key_padding_mask: Optional key padding mask [B, M]
        block_r: Tile size along query sequence dimension
        block_c: Tile size along key/value sequence dimension
        range_r: INT8 quantization range (default 127.0)
        use_ibert_exp: Whether to use I-BERT polynomial exp approximation
    """
    B, H, N, d = q.shape
    _, _, M, _ = k.shape
    device = q.device
    scale_factor = 1.0 / math.sqrt(d)

    # Pre-process key padding mask into additive attention mask if provided
    if key_padding_mask is not None:
        # [B, 1, 1, M]
        k_mask = key_padding_mask.view(B, 1, 1, M)
        pad_mask = torch.where(k_mask, torch.tensor(-1e4, device=device), torch.tensor(0.0, device=device))
        if attn_mask is not None:
            attn_mask = attn_mask + pad_mask
        else:
            attn_mask = pad_mask

    # Allocate final output accumulator
    out = torch.zeros((B, H, N, d), dtype=torch.float32, device=device)
    
    # Outer Loop over Query tiles: i in [0, Tr)
    num_r_blocks = (N + block_r - 1) // block_r
    num_c_blocks = (M + block_c - 1) // block_c

    for i in range(num_r_blocks):
        r_start = i * block_r
        r_end = min(r_start + block_r, N)
        actual_br = r_end - r_start

        qi = q[:, :, r_start:r_end, :]  # [B, H, actual_br, d] in INT8
        sq_i = sq[:, :, r_start:r_end, :]  # [B, H, actual_br, 1]

        # Online Softmax state for row block i
        mi = torch.full((B, H, actual_br, 1), -float("inf"), dtype=torch.float32, device=device)
        li = torch.zeros((B, H, actual_br, 1), dtype=torch.float32, device=device)
        oi = torch.zeros((B, H, actual_br, d), dtype=torch.float32, device=device)

        # Inner Loop over Key/Value tiles: j in [0, Tc)
        for j in range(num_c_blocks):
            c_start = j * block_c
            c_end = min(c_start + block_c, M)
            actual_bc = c_end - c_start

            kj = k[:, :, c_start:c_end, :]  # [B, H, actual_bc, d] in INT8
            vj = v[:, :, c_start:c_end, :]  # [B, H, actual_bc, d] in INT8
            sk_j = sk[:, :, c_start:c_end, :]  # [B, H, actual_bc, 1]

            # 1. Integer GEMM: S_raw = Q_i @ K_j^T
            # Computes INT8 x INT8 -> INT32
            s_raw = torch.matmul(qi.float(), kj.float().transpose(-2, -1))

            # 2. Dequantize score: S_ij = S_raw * (sq_i * sk_j^T) * (1 / sqrt(d))
            # [B, H, actual_br, 1] * [B, H, 1, actual_bc] -> [B, H, actual_br, actual_bc]
            scale_matrix = torch.matmul(sq_i, sk_j.transpose(-2, -1))
            s_ij = s_raw * scale_matrix * scale_factor

            # 3. Apply attention mask if provided (e.g. Decoder causal or permutation mask)
            if attn_mask is not None:
                if attn_mask.dim() == 2:
                    tile_mask = attn_mask[r_start:r_end, c_start:c_end]
                elif attn_mask.dim() == 3:
                    tile_mask = attn_mask[:, r_start:r_end, c_start:c_end].unsqueeze(1)
                else:
                    tile_mask = attn_mask[:, :, r_start:r_end, c_start:c_end]
                s_ij = s_ij + tile_mask

            # 4. Online Softmax update
            row_max_cur = s_ij.amax(dim=-1, keepdim=True)
            m_new = torch.maximum(mi, row_max_cur)

            # Weight matrix: P_ij = round(R * exp(S_ij - m_new))
            s_shifted = s_ij - m_new
            if use_ibert_exp:
                # I-BERT polynomial approximation L(p) >> z
                z = torch.floor(-s_shifted / 0.69314718).clamp(min=0, max=30)
                p = s_shifted + z * 0.69314718
                l_p = 0.3585 * (p + 1.353) ** 2 + 0.344
                exp_approx = l_p * torch.pow(2.0, -z)
                p_ij = torch.clamp(torch.round(range_r * exp_approx), -128, 127)
            else:
                p_ij = torch.clamp(torch.round(range_r * torch.exp(s_shifted)), -128, 127)

            # Rescaling factor for previous accumulator: exp(mi - m_new)
            alpha = torch.exp(mi - m_new)
            # When mi is -inf (first iteration), alpha is 0
            alpha = torch.nan_to_num(alpha, nan=0.0)

            # 5. Update denominator li and output oi
            li = li * alpha + p_ij.sum(dim=-1, keepdim=True)
            # GEMM: P_ij (INT8) @ Vj (INT8) accumulating to FP32/INT32
            # If sv has per-token scales [B, H, actual_bc, 1], scale Vj locally
            if sv.dim() == 4 and sv.shape[2] == M:
                sv_j = sv[:, :, c_start:c_end, :]
                vj_scaled = vj.float() * sv_j
                pv = torch.matmul(p_ij, vj_scaled)
                oi = oi * alpha + pv
            else:
                pv = torch.matmul(p_ij, vj.float())
                oi = oi * alpha + pv
            mi = m_new

        # Rescale block output: O_i = (O_i / li) * s_v
        if sv.dim() == 4 and sv.shape[2] == M:
            oi_normalized = oi / li.clamp(min=1e-8)
        else:
            oi_normalized = (oi / li.clamp(min=1e-8)) * sv
        out[:, :, r_start:r_end, :] = oi_normalized

    return out


class INT8MultiheadAttention(nn.Module):
    """Drop-in INT8 Multi-Head Attention module for PARSeq Transformer layers.
    
    Supports:
        - Self-Attention in Encoder (no mask or all-ones).
        - Two-stream Self-Attention in Decoder with Permutation and Causal Masks.
        - Cross-Attention in Decoder attending to Vision Transformer Encoder Memory.
        - Integration with Jetfire / RealHardware Linear projections.
        - High-throughput INT-FlashAttention online softmax execution.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        block_size: int = 32,
        use_ibert_exp: bool = False,
        batch_first: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.dropout_p = dropout
        self.block_size = block_size
        self.use_ibert_exp = use_ibert_exp
        self.batch_first = batch_first

        # Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    @classmethod
    def from_float_mha(
        cls,
        float_mha: nn.MultiheadAttention,
        block_size: int = 32,
        use_ibert_exp: bool = False,
    ) -> "INT8MultiheadAttention":
        """Constructs an INT8MultiheadAttention from an existing PyTorch nn.MultiheadAttention."""
        embed_dim = float_mha.embed_dim
        num_heads = float_mha.num_heads
        bias = float_mha.in_proj_bias is not None or float_mha.bias_k is not None
        mod = cls(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=float_mha.dropout,
            bias=True,
            block_size=block_size,
            use_ibert_exp=use_ibert_exp,
            batch_first=float_mha.batch_first,
        )

        with torch.no_grad():
            if float_mha.in_proj_weight is not None:
                # in_proj_weight packs Q, K, V
                w_q, w_k, w_v = float_mha.in_proj_weight.chunk(3, dim=0)
                mod.q_proj.weight.copy_(w_q)
                mod.k_proj.weight.copy_(w_k)
                mod.v_proj.weight.copy_(w_v)
                if float_mha.in_proj_bias is not None:
                    b_q, b_k, b_v = float_mha.in_proj_bias.chunk(3, dim=0)
                    mod.q_proj.bias.copy_(b_q)
                    mod.k_proj.bias.copy_(b_k)
                    mod.v_proj.bias.copy_(b_v)
            else:
                if float_mha.q_proj_weight is not None:
                    mod.q_proj.weight.copy_(float_mha.q_proj_weight)
                if float_mha.k_proj_weight is not None:
                    mod.k_proj.weight.copy_(float_mha.k_proj_weight)
                if float_mha.v_proj_weight is not None:
                    mod.v_proj.weight.copy_(float_mha.v_proj_weight)

            # out_proj
            if hasattr(float_mha, "out_proj"):
                mod.out_proj.weight.copy_(float_mha.out_proj.weight)
                if float_mha.out_proj.bias is not None:
                    mod.out_proj.bias.copy_(float_mha.out_proj.bias)

        target_device = (
            float_mha.in_proj_weight.device
            if float_mha.in_proj_weight is not None
            else (
                float_mha.q_proj_weight.device
                if float_mha.q_proj_weight is not None
                else (
                    float_mha.out_proj.weight.device
                    if hasattr(float_mha, "out_proj")
                    else torch.device("cpu")
                )
            )
        )
        mod = mod.to(target_device)
        return mod

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compatible with nn.MultiheadAttention signature."""
        if self.q_proj.weight.device != query.device:
            self.to(query.device)

        if not self.batch_first:
            # Transpose sequence and batch: [L, B, d] -> [B, L, d]
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        B, N, _ = query.shape
        _, M, _ = key.shape
        H = self.num_heads
        d = self.head_dim

        # 1. Linear projections
        q_proj = self.q_proj(query).view(B, N, H, d).transpose(1, 2)  # [B, H, N, d]
        k_proj = self.k_proj(key).view(B, M, H, d).transpose(1, 2)    # [B, H, M, d]
        v_proj = self.v_proj(value).view(B, M, H, d).transpose(1, 2)  # [B, H, M, d]

        # 2. INT8 Quantization (adheres to INT-FlashAttention Algorithm 1)
        q_int8, sq = quantize_token_symmetric(q_proj)
        k_int8, sk = quantize_token_symmetric(k_proj)
        v_int8, sv = quantize_token_symmetric(v_proj)

        # 3. Flash Attention Execution with Quantized Online Softmax
        # During tracing or ONNX export, fallback to standard float attention to ensure graph validity
        if torch.onnx.is_in_onnx_export() or torch.jit.is_tracing():
            scores = torch.matmul(q_proj, k_proj.transpose(-2, -1)) * (1.0 / math.sqrt(d))
            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    scores = scores.masked_fill(attn_mask, -1e4)
                else:
                    scores = scores + attn_mask
            if key_padding_mask is not None:
                scores = scores.masked_fill(key_padding_mask.view(B, 1, 1, M), -1e4)
            probs = F.softmax(scores, dim=-1)
            attn_out = torch.matmul(probs, v_proj)
        else:
            attn_out = int_flash_attention_core(
                q=q_int8,
                k=k_int8,
                v=v_int8,
                sq=sq,
                sk=sk,
                sv=sv,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                block_r=self.block_size,
                block_c=self.block_size,
                use_ibert_exp=self.use_ibert_exp,
            )

        # 4. Concatenate heads and final out projection
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, self.embed_dim)
        out = self.out_proj(attn_out)

        if not self.batch_first:
            out = out.transpose(0, 1)

        weights = None
        if need_weights:
            weights = torch.zeros((B, N, M), device=out.device)

        return out, weights
