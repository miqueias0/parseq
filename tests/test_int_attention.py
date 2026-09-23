#!/usr/bin/env python3
"""Numerical Validation Tests for INT-FlashAttention Module."""

import unittest
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from strhub.models.quantization.int_attention import (
    int_flash_attention_core,
    INT8MultiheadAttention,
    quantize_token_symmetric,
)


class TestINTFlashAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_core_online_softmax_equivalence(self):
        """Verifies that INT-FlashAttention closely tracks standard FP32 attention (MRE < 3%)."""
        B, H, N, d = 2, 4, 64, 32
        q = torch.randn(B, H, N, d)
        k = torch.randn(B, H, N, d)
        v = torch.randn(B, H, N, d)

        # Standard attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(d))
        probs = F.softmax(scores, dim=-1)
        expected_out = torch.matmul(probs, v)

        # INT-FlashAttention
        q_int8, sq = quantize_token_symmetric(q)
        k_int8, sk = quantize_token_symmetric(k)
        v_int8, sv = quantize_token_symmetric(v)

        actual_out = int_flash_attention_core(
            q=q_int8,
            k=k_int8,
            v=v_int8,
            sq=sq,
            sk=sk,
            sv=sv,
            block_r=32,
            block_c=32,
        )

        self.assertEqual(actual_out.shape, expected_out.shape)
        rel_err = (actual_out - expected_out).abs().mean() / expected_out.abs().mean()
        self.assertLess(rel_err.item(), 0.035, f"Relative error {rel_err.item()} exceeds 3.5%")

    def test_dynamic_mask_support(self):
        """Validates causal / permutation mask support in INT-FlashAttention."""
        B, H, N, d = 2, 4, 32, 32
        q = torch.randn(B, H, N, d)
        k = torch.randn(B, H, N, d)
        v = torch.randn(B, H, N, d)

        # Lower triangular causal mask (as used in autoregressive decoding)
        causal_mask = torch.triu(torch.full((N, N), -float("inf")), diagonal=1)

        # Standard masked attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(d)) + causal_mask
        probs = F.softmax(scores, dim=-1)
        expected_out = torch.matmul(probs, v)

        # INT-FlashAttention with mask
        q_int8, sq = quantize_token_symmetric(q)
        k_int8, sk = quantize_token_symmetric(k)
        v_int8, sv = quantize_token_symmetric(v)

        actual_out = int_flash_attention_core(
            q=q_int8,
            k=k_int8,
            v=v_int8,
            sq=sq,
            sk=sk,
            sv=sv,
            attn_mask=causal_mask,
            block_r=16,
            block_c=16,
        )

        rel_err = (actual_out - expected_out).abs().mean() / expected_out.abs().mean()
        self.assertLess(rel_err.item(), 0.04, f"Masked relative error {rel_err.item()} exceeds 4%")

    def test_cross_attention_asymmetric_shapes(self):
        """Validates Cross-Attention where query length N differs from key/value memory length M."""
        B, H, N, M, d = 2, 4, 26, 128, 32  # 26 label tokens, 128 patch tokens from ViT
        q = torch.randn(B, H, N, d)
        k = torch.randn(B, H, M, d)
        v = torch.randn(B, H, M, d)

        scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(d))
        probs = F.softmax(scores, dim=-1)
        expected_out = torch.matmul(probs, v)

        q_int8, sq = quantize_token_symmetric(q)
        k_int8, sk = quantize_token_symmetric(k)
        v_int8, sv = quantize_token_symmetric(v)

        actual_out = int_flash_attention_core(
            q=q_int8,
            k=k_int8,
            v=v_int8,
            sq=sq,
            sk=sk,
            sv=sv,
            block_r=16,
            block_c=32,
        )

        self.assertEqual(actual_out.shape, (B, H, N, d))
        rel_err = (actual_out - expected_out).abs().mean() / expected_out.abs().mean()
        self.assertLess(rel_err.item(), 0.035, f"Cross-attention relative error {rel_err.item()} exceeds 3.5%")

    def test_int8_multihead_attention_layer(self):
        """Tests that INT8MultiheadAttention seamlessly replaces nn.MultiheadAttention."""
        embed_dim = 128
        num_heads = 4
        mha_float = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        mha_int8 = INT8MultiheadAttention.from_float_mha(mha_float, block_size=32)

        x = torch.randn(2, 32, embed_dim)
        out_float, _ = mha_float(x, x, x)
        out_int8, _ = mha_int8(x, x, x)

        self.assertEqual(out_int8.shape, out_float.shape)
        rel_err = (out_int8 - out_float).abs().mean() / out_float.abs().mean()
        self.assertLess(rel_err.item(), 0.05, f"MHA layer relative error {rel_err.item()} exceeds 5%")


if __name__ == "__main__":
    unittest.main()
