import pytest
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from strhub.quant.quant_utils import (
    quantize_symmetric,
    dequantize_symmetric,
    quantize_naive,
    compute_sqnr,
    compute_mse,
    compute_cosine_similarity,
    compute_saturation_stats,
)
from strhub.quant.integer_gelu import IBERTGELU, IViTGELU, IPTQDataAwarePolyGELU, GELUFP32
from strhub.quant.integer_softmax import IBERTSoftmax, IViTShiftmax, IPTQBitSoftmax, SoftmaxFP32
from strhub.quant.integer_layernorm import IBERTLayerNorm, IPTQLayerNorm, LayerNormFP32, integer_sqrt_newton
from strhub.quant.unified_metric import compute_unified_metric, softplus
from strhub.models.parseq.model import PARSeq
from strhub.models.parseq.quantized_parseq import create_model_variant


def test_quantize_symmetric():
    x = torch.tensor([-10.0, -5.0, 0.0, 5.0, 10.0])
    q, scale = quantize_symmetric(x, bits=8)
    assert q.min() >= -128
    assert q.max() <= 127
    deq = dequantize_symmetric(q, scale)
    assert compute_mse(x, deq) < 0.05
    assert compute_sqnr(x, deq) > 20.0


def test_quantize_naive_negative_control():
    # Naive INT8 truncates and clamps without scale adaptation
    x = torch.tensor([-500.0, -2.5, 0.5, 3.8, 300.0])
    q, scale = quantize_naive(x, bits=8)
    assert q.min() == -128
    assert q.max() == 127
    assert scale.item() == 1.0


def test_gelu_candidates_finite_and_bounded():
    x = torch.linspace(-4.0, 4.0, 100)
    gelu_fp = GELUFP32()
    gelu_ibert = IBERTGELU()
    gelu_ivit = IViTGELU()
    gelu_iptq = IPTQDataAwarePolyGELU()

    out_fp = gelu_fp(x)
    out_ibert = gelu_ibert(x)
    out_ivit = gelu_ivit(x)
    out_iptq = gelu_iptq(x)

    for out, name in [(out_ibert, "i-bert"), (out_ivit, "i-vit"), (out_iptq, "iptq")]:
        assert torch.isfinite(out).all(), f"{name} produced non-finite values"
        assert compute_cosine_similarity(out_fp, out) > 0.98, f"{name} cosine similarity below 0.98"


def test_softmax_candidates_probability_distribution():
    x = torch.randn(4, 10)
    sm_fp = SoftmaxFP32(dim=-1)
    sm_ibert = IBERTSoftmax(dim=-1)
    sm_ivit = IViTShiftmax(dim=-1)
    sm_iptq = IPTQBitSoftmax(dim=-1)

    out_fp = sm_fp(x)
    out_ibert = sm_ibert(x)
    out_ivit = sm_ivit(x)
    out_iptq = sm_iptq(x)

    for out, name in [(out_ibert, "i-bert"), (out_ivit, "i-vit"), (out_iptq, "iptq")]:
        assert torch.isfinite(out).all(), f"{name} produced non-finite values"
        assert (out >= 0.0).all(), f"{name} produced negative probabilities"
        sums = torch.sum(out, dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-3), f"{name} probabilities do not sum to 1"
        assert compute_cosine_similarity(out_fp, out) > 0.95, f"{name} attention similarity below 0.95"


def test_integer_layernorm_and_sqrt():
    x = torch.randn(2, 5, 64)
    ln_fp = LayerNormFP32(64)
    ln_ibert = IBERTLayerNorm(64)

    out_fp = ln_fp(x)
    out_ibert = ln_ibert(x)

    assert torch.isfinite(out_ibert).all()
    assert compute_cosine_similarity(out_fp, out_ibert) > 0.95

    # Test Newton integer square root convergence
    val = torch.tensor([4.0, 9.0, 16.0, 100.0, 10000.0])
    sqrt_res = integer_sqrt_newton(val, max_iters=4)
    assert torch.allclose(sqrt_res, torch.sqrt(val), atol=1e-2)


def test_unified_metric_properties():
    # Omega increases with higher SQNR (Q) and lower perturbation (P) and lower cost (C)
    q_high, q_low = 30.0, 10.0
    p_low, p_high = 0.01, 1.0
    c_low, c_high = 5.0, 20.0

    omega_best = compute_unified_metric(q_high, p_low, c_low)
    omega_worst = compute_unified_metric(q_low, p_high, c_high)
    assert omega_best > omega_worst


def test_strict_nar_configuration():
    """AC2 [CRITICAL]: Verify Non-Autoregressive (NAR) decode_ar=False and refine_iters=0."""
    model = PARSeq(
        num_tokens=38,
        max_label_length=7,
        img_size=[32, 128],
        patch_size=[4, 8],
        embed_dim=384,
        enc_num_heads=6,
        enc_mlp_ratio=4,
        enc_depth=2, # small for fast test
        dec_num_heads=6,
        dec_mlp_ratio=4,
        dec_depth=1,
        decode_ar=False,
        refine_iters=0,
        dropout=0.0,
    )
    assert model.decode_ar is False
    assert model.refine_iters == 0

    # Build M1 NAR
    m1 = create_model_variant("m1", model)
    assert m1.decode_ar is False
    assert m1.refine_iters == 0

    # Build M5 Integer-Only PTQ
    m5 = create_model_variant("m5", model)
    assert m5.decode_ar is False
    assert m5.refine_iters == 0


def test_dynamic_batching():
    """Verify dynamic batching with batch sizes 1, 4, 16, 32 without dimension broadcast crash."""
    from tools.export_onnx import ONNXExportWrapper
    from unittest.mock import MagicMock
    model = PARSeq(
        num_tokens=38,
        max_label_length=7,
        img_size=[32, 128],
        patch_size=[4, 8],
        embed_dim=384,
        enc_num_heads=6,
        enc_mlp_ratio=4,
        enc_depth=1,
        dec_num_heads=6,
        dec_mlp_ratio=4,
        dec_depth=1,
        decode_ar=False,
        refine_iters=0,
        dropout=0.0,
    )
    tokenizer = MagicMock()
    tokenizer.bos_id = 0
    wrapper = ONNXExportWrapper(model, tokenizer)
    for bs in [1, 4, 16, 32]:
        x = torch.randn(bs, 3, 32, 128)
        out = wrapper(x)
        assert out.shape == (bs, 8, 36)


def test_fusion_flags():
    """Verify modular kernel fusion flags (fuse_mha, fuse_mlp, fuse_layernorm)."""
    model = PARSeq(
        num_tokens=38,
        max_label_length=7,
        img_size=[32, 128],
        patch_size=[4, 8],
        embed_dim=384,
        enc_num_heads=6,
        enc_mlp_ratio=4,
        enc_depth=2,
        dec_num_heads=6,
        dec_mlp_ratio=4,
        dec_depth=1,
        decode_ar=False,
        refine_iters=0,
        dropout=0.0,
    )

    # Test baseline M5 without fusion
    m5_baseline = create_model_variant("m5", model, fuse_mha=False, fuse_mlp=False, fuse_layernorm=False)
    assert m5_baseline.encoder.blocks[0].attn.fuse_mha is False

    # Test M5 with modular fusion flags
    m5_fused = create_model_variant("m5", model, fuse_mha=True, fuse_mlp=True, fuse_layernorm=True)
    assert m5_fused.encoder.blocks[0].attn.fuse_mha is True
    assert isinstance(m5_fused.encoder.blocks[0].mlp.act, GELUFP32)
    assert isinstance(m5_fused.encoder.blocks[0].norm1, LayerNormFP32)

    # Forward pass on fused variant
    x = torch.randn(2, 3, 32, 128)
    encoded = m5_fused.encode(x)
    assert encoded.shape[0] == 2


def test_int_flashattention_algorithm1():
    """Verify INT-FlashAttention Algorithm 1 fidelity (arXiv:2409.16997v2).
    Checks quantization error against exact scaled dot-product attention
    under normal N(0, 1) and uniform U(-0.5, 0.5) distributions (Tables 1 & 2 of the paper).
    """
    from strhub.quant.int_flashattention import int_flashattention_forward

    B, H, N, d = 2, 6, 128, 64
    torch.manual_seed(42)

    # 1. Normal distributed activations
    q_norm = torch.randn(B, H, N, d)
    k_norm = torch.randn(B, H, N, d)
    v_norm = torch.randn(B, H, N, d)

    out_exact = F.scaled_dot_product_attention(q_norm, k_norm, v_norm)
    out_int_fa = int_flashattention_forward(q_norm, k_norm, v_norm, block_r=64, block_c=64, bits=8)

    assert out_int_fa.shape == out_exact.shape
    assert torch.isfinite(out_int_fa).all()

    # Relative error (MRE) within paper's bound (< 5%)
    mre = (out_int_fa - out_exact).abs().mean().item() / out_exact.abs().mean().item()
    assert mre < 0.05, f"MRE {mre:.4f} exceeded upper bound 0.05"

    # Cosine similarity must be extremely high (> 0.98)
    cos_sim = F.cosine_similarity(out_int_fa.flatten(), out_exact.flatten(), dim=0).item()
    assert cos_sim > 0.98, f"Cosine similarity {cos_sim:.4f} below 0.98"

    # 2. Uniform distributed activations
    q_unif = torch.empty(B, H, N, d).uniform_(-0.5, 0.5)
    k_unif = torch.empty(B, H, N, d).uniform_(-0.5, 0.5)
    v_unif = torch.empty(B, H, N, d).uniform_(-0.5, 0.5)

    out_exact_unif = F.scaled_dot_product_attention(q_unif, k_unif, v_unif)
    out_int_fa_unif = int_flashattention_forward(q_unif, k_unif, v_unif, block_r=64, block_c=64, bits=8)

    cos_sim_unif = F.cosine_similarity(out_int_fa_unif.flatten(), out_exact_unif.flatten(), dim=0).item()
    assert cos_sim_unif > 0.98, f"Cosine similarity for uniform {cos_sim_unif:.4f} below 0.98"


def test_int_flashattention_model_integration():
    """Verify integration of INT-FlashAttention flag into PARSeq model variants and dynamic batching."""
    from unittest.mock import MagicMock
    from tools.export_onnx import ONNXExportWrapper

    model = PARSeq(
        num_tokens=38,
        max_label_length=7,
        img_size=[32, 128],
        patch_size=[4, 8],
        embed_dim=384,
        enc_num_heads=6,
        enc_mlp_ratio=4,
        enc_depth=2,
        dec_num_heads=6,
        dec_mlp_ratio=4,
        dec_depth=1,
        decode_ar=False,
        refine_iters=0,
        dropout=0.0,
    )

    # Build M5 with INT-FlashAttention
    m5_int_fa = create_model_variant("m5", model, use_int_flashattention=True)
    assert m5_int_fa.encoder.blocks[0].attn.use_int_flashattention is True
    assert m5_int_fa.encoder.blocks[0].attn.int_flash_attn is not None

    tokenizer = MagicMock()
    tokenizer.bos_id = 0
    wrapper = ONNXExportWrapper(m5_int_fa, tokenizer)

    # Test dynamic batch forward pass
    for bs in [1, 3, 7]:
        x = torch.randn(bs, 3, 32, 128)
        encoded = m5_int_fa.encode(x)
        assert encoded.shape == (bs, 128, 384)
        assert torch.isfinite(encoded).all()

        out = wrapper(x)
        assert out.shape == (bs, 8, 36)
        assert torch.isfinite(out).all()

