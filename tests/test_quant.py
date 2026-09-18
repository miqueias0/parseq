import pytest
import math
import torch
import torch.nn as nn

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
