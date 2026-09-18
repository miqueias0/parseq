import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def int_flashattention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    attn_mask: Optional[torch.Tensor] = None,
    block_r: int = 64,
    block_c: int = 64,
    bits: int = 8,
) -> torch.Tensor:
    """Implements INT-FlashAttention forward pass strictly adhering to Algorithm 1
    of arXiv:2409.16997v2 (Chen et al., Peking University / Baichuan, 2024).

    Workflow:
    1. Linear symmetric token-level quantization of Q and K with scaling factors SQ, SK.
    2. Tensor-level quantization of V with scalar SV.
    3. Block-tiled computation (Br x Bc) directly emulating SRAM-resident attention:
       - GEMM-1: S_i^(j) = scale * diag(SQ_i) * (Q_int_i @ K_int_j^T) * diag(SK_j)
       - Online Softmax: tracks running row-wise maximum m_i and exponential sum l_i
       - INT8 Attention Matrix: P_i^(j) = round(R * exp(S_i^(j) - m_i)) in [0, R]
       - GEMM-2: O_i^(j) = diag(exp(m_i^(j-1) - m_i^(j))) * O_i^(j-1) + P_i^(j) @ V_int_j
    4. Online dequantization and normalization: O_i = diag(l_i)^(-1) * O_i * SV.

    Args:
        q: Query tensor of shape (B, num_heads, N, head_dim)
        k: Key tensor of shape (B, num_heads, N, head_dim)
        v: Value tensor of shape (B, num_heads, N, head_dim)
        scale: Scaling factor (default: 1.0 / sqrt(head_dim))
        attn_mask: Optional attention mask
        block_r: Row block size Br (default: 64)
        block_c: Column block size Bc (default: 64)
        bits: Bit-width for quantization (default: 8)

    Returns:
        Output tensor O of shape (B, num_heads, N, head_dim)
    """
    B, H, N, d = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(d)

    R = float((1 << (bits - 1)) - 1)  # 127 for INT8
    device = q.device
    dtype = q.dtype

    # Step 1: Token-level symmetric quantization for Q and K (Section 3.2, p. 4)
    # SQ = rowmax(|Q|) / R, SK = rowmax(|K|) / R
    q_max = q.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    k_max = k.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    s_q = q_max / R  # (B, H, N, 1)
    s_k = k_max / R  # (B, H, N, 1)

    q_int = torch.clamp(torch.round(q / s_q), -R - 1, R)
    k_int = torch.clamp(torch.round(k / s_k), -R - 1, R)

    # Step 2: Tensor-level quantization for V (Section 3.2, p. 4)
    # SV = max(|V|) / R
    v_max = v.abs().amax().clamp(min=1e-8)
    s_v = v_max / R
    v_int = torch.clamp(torch.round(v / s_v), -R - 1, R)

    # List to collect output row blocks
    O_blocks = []

    # Determine tiling parameters
    Tr = (N + block_r - 1) // block_r
    Tc = (N + block_c - 1) // block_c

    # Outer loop: iterate over row blocks of Q (Line 4-5 of Algorithm 1)
    for i in range(Tr):
        r_start = i * block_r
        r_end = min((i + 1) * block_r, N)
        actual_br = r_end - r_start

        q_i = q_int[:, :, r_start:r_end, :]  # (B, H, actual_br, d)
        s_qi = s_q[:, :, r_start:r_end, :]   # (B, H, actual_br, 1)

        O_i = None
        l_i = None
        m_i = None

        # Inner loop: iterate over column blocks of K and V (Line 7-8 of Algorithm 1)
        for j in range(Tc):
            c_start = j * block_c
            c_end = min((j + 1) * block_c, N)
            actual_bc = c_end - c_start

            k_j = k_int[:, :, c_start:c_end, :]  # (B, H, actual_bc, d)
            v_j = v_int[:, :, c_start:c_end, :]  # (B, H, actual_bc, d)
            s_kj = s_k[:, :, c_start:c_end, :]   # (B, H, actual_bc, 1)

            # Line 9: S_i^(j) = scale * diag(SQ_i) * (Q_i @ K_j^T) * diag(SK_j)
            gemm1 = torch.matmul(q_i, k_j.transpose(-2, -1))  # (B, H, actual_br, actual_bc)
            S_ij = (gemm1 * s_qi * s_kj.transpose(-2, -1)) * scale

            if attn_mask is not None:
                mask_slice = attn_mask[..., r_start:r_end, c_start:c_end]
                S_ij = S_ij + mask_slice

            if j == 0:
                # Initialization for first block (no alpha rescaling needed)
                m_i = S_ij.amax(dim=-1, keepdim=True)
                exp_s = torch.exp(S_ij - m_i)
                P_ij = torch.clamp(torch.round(R * exp_s), 0.0, R)
                l_i = P_ij.sum(dim=-1, keepdim=True)
                O_i = torch.matmul(P_ij, v_j)
            else:
                # Line 10: m_i^(j) = max(m_i^(j-1), rowmax(S_i^(j)))
                m_ij = S_ij.amax(dim=-1, keepdim=True)
                m_new = torch.maximum(m_i, m_ij)

                # Line 11: P_i^(j) = round(R * exp(S_i^(j) - m_new)) in INT8
                exp_s = torch.exp(S_ij - m_new)
                P_ij = torch.clamp(torch.round(R * exp_s), 0.0, R)

                # Line 12: l_i^(j) = exp(m_old - m_new) * l_i^(j-1) + rowsum(P_i^(j))
                alpha = torch.exp(m_i - m_new)
                l_i = alpha * l_i + P_ij.sum(dim=-1, keepdim=True)

                # Line 13: O_i^(j) = diag(alpha) * O_i^(j-1) + P_ij @ V_j
                gemm2 = torch.matmul(P_ij, v_j)
                O_i = alpha * O_i + gemm2

                # Update running maximum
                m_i = m_new

        # Line 16: O_i = diag(l_i)^(-1) * O_i * SV (Dequantization to float)
        norm_factor = (l_i.clamp(min=1e-8)) ** -1
        O_i = (O_i * norm_factor) * s_v
        O_blocks.append(O_i)

    # Line 17 & 19: Concatenate row blocks to form output O
    if len(O_blocks) == 1:
        O = O_blocks[0]
    else:
        O = torch.cat(O_blocks, dim=2)

    return O.to(dtype=dtype)


class INTFlashAttention(nn.Module):
    """INT-FlashAttention Module for Transformer Attention blocks.
    Compatible with FlashAttention forward workflow, operating fully in INT8.

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
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.block_r = block_r
        self.block_c = block_c
        self.bits = bits

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Executes INT-FlashAttention Algorithm 1.
        q, k, v are of shape (B, num_heads, N, head_dim).
        """
        return int_flashattention_forward(
            q=q,
            k=k,
            v=v,
            scale=self.scale,
            attn_mask=attn_mask,
            block_r=self.block_r,
            block_c=self.block_c,
            bits=self.bits,
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
    ):
        super().__init__()
        self.attn = original_attn
        self.int_flash_attn = INTFlashAttention(
            embed_dim=self.attn.attn_dim if hasattr(self.attn, "attn_dim") else self.attn.qkv.out_features // 3,
            num_heads=self.attn.num_heads,
            block_r=block_r,
            block_c=block_c,
            bits=bits,
        )

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if "attn" in self.__dict__:
                return getattr(self.attn, name)
            raise

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, is_causal: bool = False) -> torch.Tensor:
        B, N, C = x.shape
        # QKV linear projection
        qkv = self.attn.qkv(x).reshape(B, N, 3, self.attn.num_heads, self.attn.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.attn.q_norm(q), self.attn.k_norm(k)

        # Fused INT8 FlashAttention forward pass (Algorithm 1)
        x = self.int_flash_attn(q, k, v, attn_mask=attn_mask)

        x = x.transpose(1, 2).reshape(B, N, self.attn.attn_dim)
        x = self.attn.norm(x)
        x = self.attn.proj(x)
        x = self.attn.proj_drop(x)
        return x
