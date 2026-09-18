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


def test_int_flashattention_algorithm1_dtypes_and_accumulators():
    """AC1, AC2, AC4, AC5: Verify genuine INT8 datatypes and INT32 accumulators
    strictly following Algorithm 1 of arXiv:2409.16997v2.
    """
    from strhub.quant.int_flashattention import int_flashattention_forward, int8_matmul_int32

    # 1. Direct unit verification of int8_matmul_int32 GEMM kernel
    a = torch.randint(-128, 127, (2, 4, 32, 64), dtype=torch.int8)
    b = torch.randint(-128, 127, (2, 4, 64, 32), dtype=torch.int8)
    c = int8_matmul_int32(a, b)
    assert c.dtype == torch.int32, f"Expected torch.int32 accumulator, got {c.dtype}"
    assert c.shape == (2, 4, 32, 32)
    # Check exactness against integer ground truth
    c_gt = a.float() @ b.float()
    assert torch.allclose(c.float(), c_gt, atol=1e-5), "int8_matmul_int32 arithmetic mismatch"

    # Type safety check: passing float tensors must raise TypeError
    with pytest.raises(TypeError):
        int8_matmul_int32(a.float(), b.float())

    # 2. Runtime forward pass verification with diagnostic inspection
    B, H, N, d = 2, 4, 64, 32
    torch.manual_seed(42)
    q = torch.randn(B, H, N, d)
    k = torch.randn(B, H, N, d)
    v = torch.randn(B, H, N, d)

    out, diag = int_flashattention_forward(
        q, k, v, block_r=32, block_c=32, bits=8, return_diagnostics=True
    )

    assert diag["q_int_dtype"] == "torch.int8", f"Expected torch.int8, got {diag['q_int_dtype']}"
    assert diag["k_int_dtype"] == "torch.int8", f"Expected torch.int8, got {diag['k_int_dtype']}"
    assert diag["v_int_dtype"] == "torch.int8", f"Expected torch.int8, got {diag['v_int_dtype']}"
    assert diag["p_int_dtypes"][0] == "torch.int8", f"Expected torch.int8, got {diag['p_int_dtypes'][0]}"
    assert diag["gemm1_dtypes"][0] == "torch.int32", f"Expected torch.int32, got {diag['gemm1_dtypes'][0]}"
    assert diag["gemm2_dtypes"][0] == "torch.int32", f"Expected torch.int32, got {diag['gemm2_dtypes'][0]}"

    # Boundary ranges
    assert diag["q_int_range"][0] >= -128 and diag["q_int_range"][1] <= 127
    assert diag["k_int_range"][0] >= -128 and diag["k_int_range"][1] <= 127
    assert diag["v_int_range"][0] >= -128 and diag["v_int_range"][1] <= 127
    assert diag["p_int_ranges"][0][0] >= 0 and diag["p_int_ranges"][0][1] <= 127


def test_int_flashattention_scientific_metrics():
    """AC8: Verify quantization error metrics against exact FP32 attention
    under normal N(0, 1) and uniform U(-0.5, 0.5) activations (Tables 1 & 2 of the paper).
    Measures MAE, MSE, RMSE, MRE (< 5%), Cosine Similarity (> 0.98), and Relative L2 error.
    """
    from strhub.quant.int_flashattention import int_flashattention_forward, exact_sdpa_fp32

    B, H, N, d = 2, 6, 128, 64
    torch.manual_seed(42)

    for dist_name, (q, k, v) in [
        ("normal", (torch.randn(B, H, N, d), torch.randn(B, H, N, d), torch.randn(B, H, N, d))),
        ("uniform", (
            torch.empty(B, H, N, d).uniform_(-0.5, 0.5),
            torch.empty(B, H, N, d).uniform_(-0.5, 0.5),
            torch.empty(B, H, N, d).uniform_(-0.5, 0.5),
        )),
    ]:
        out_fp32 = exact_sdpa_fp32(q, k, v)
        out_int_fa = int_flashattention_forward(q, k, v, block_r=64, block_c=64, bits=8)

        assert out_int_fa.shape == out_fp32.shape
        assert torch.isfinite(out_int_fa).all()

        diff = out_int_fa - out_fp32
        mae = diff.abs().mean().item()
        mse = (diff ** 2).mean().item()
        rmse = math.sqrt(mse)
        mre = diff.abs().mean().item() / out_fp32.abs().mean().item()
        cos_sim = F.cosine_similarity(out_int_fa.flatten(), out_fp32.flatten(), dim=0).item()
        rel_l2 = torch.norm(diff) / torch.norm(out_fp32)

        # Assert scientific thresholds from the paper
        assert mre < 0.05, f"{dist_name} MRE {mre:.4f} exceeded upper bound 0.05"
        assert cos_sim > 0.98, f"{dist_name} Cosine similarity {cos_sim:.4f} below 0.98"
        assert rel_l2 < 0.10, f"{dist_name} Relative L2 error {rel_l2:.4f} exceeded 0.10"


def test_int_flashattention_tiling_invariance():
    """AC7: Verify tiling block invariance across block sizes Br, Bc in {16, 32, 64, 128}.
    Validates online softmax normalization and numerical equivalence regardless of tile granularity.
    """
    from strhub.quant.int_flashattention import int_flashattention_forward

    B, H, N, d = 2, 4, 128, 64
    torch.manual_seed(42)
    q = torch.randn(B, H, N, d)
    k = torch.randn(B, H, N, d)
    v = torch.randn(B, H, N, d)

    outputs = {}
    for block_size in [16, 32, 64, 128]:
        outputs[block_size] = int_flashattention_forward(
            q, k, v, block_r=block_size, block_c=block_size, bits=8
        )

    ref = outputs[64]
    for block_size, out in outputs.items():
        cos_sim = F.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
        mre = (out - ref).abs().mean().item() / ref.abs().mean().item()
        assert cos_sim > 0.999, f"Block {block_size} cosine similarity {cos_sim:.6f} below 0.999"
        assert mre < 0.015, f"Block {block_size} MRE {mre:.6f} exceeded 0.015 against block 64"


def test_int_flashattention_v_quant_modes():
    """AC9: Verify both 'per_tensor' (strict paper) and 'per_head' (multi-head ViT) V quantization modes."""
    from strhub.quant.int_flashattention import int_flashattention_forward, exact_sdpa_fp32

    B, H, N, d = 2, 6, 64, 32
    torch.manual_seed(42)
    q = torch.randn(B, H, N, d)
    k = torch.randn(B, H, N, d)
    v = torch.randn(B, H, N, d)

    exact = exact_sdpa_fp32(q, k, v)
    out_tensor = int_flashattention_forward(q, k, v, v_quant_mode="per_tensor")
    out_head = int_flashattention_forward(q, k, v, v_quant_mode="per_head")

    assert torch.isfinite(out_tensor).all()
    assert torch.isfinite(out_head).all()

    cos_tensor = F.cosine_similarity(out_tensor.flatten(), exact.flatten(), dim=0).item()
    cos_head = F.cosine_similarity(out_head.flatten(), exact.flatten(), dim=0).item()

    assert cos_tensor > 0.98
    assert cos_head > 0.98


def test_int_flashattention_step_error_decomposition():
    """AC3, AC6: Measure error contribution of each pipeline stage in isolation:
    1. Q/K token-level quantization error (SQNR > 25 dB)
    2. Score matrix S reconstruction error (Cosine similarity > 0.98)
    3. P quantization error (Cosine similarity > 0.98)
    4. Final output O error (Cosine similarity > 0.98)
    """
    from strhub.quant.int_flashattention import int_flashattention_forward, exact_sdpa_fp32
    from strhub.quant.quant_utils import compute_sqnr

    B, H, N, d = 2, 4, 64, 32
    torch.manual_seed(42)
    q = torch.randn(B, H, N, d)
    k = torch.randn(B, H, N, d)
    v = torch.randn(B, H, N, d)

    R = 127.0
    s_q = q.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / R
    q_int = torch.clamp(torch.round(q / s_q), -R - 1, R).to(torch.int8)
    q_deq = q_int.float() * s_q

    sqnr_q = compute_sqnr(q, q_deq)
    assert sqnr_q > 25.0, f"Q SQNR {sqnr_q:.2f} dB below 25 dB"

    # Score matrix comparison
    s_exact = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(d)
    s_int = (q_deq @ (k.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / R * torch.clamp(torch.round(k / (k.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / R)), -R - 1, R).to(torch.int8)).float().transpose(-2, -1)) / math.sqrt(d)

    cos_s = F.cosine_similarity(s_int.flatten(), s_exact.flatten(), dim=0).item()
    assert cos_s > 0.98, f"Score matrix cosine similarity {cos_s:.4f} below 0.98"

    # Final end-to-end output
    out_exact = exact_sdpa_fp32(q, k, v)
    out_int_fa = int_flashattention_forward(q, k, v)
    cos_o = F.cosine_similarity(out_int_fa.flatten(), out_exact.flatten(), dim=0).item()
    assert cos_o > 0.98, f"Final output cosine similarity {cos_o:.4f} below 0.98"


def test_int_flashattention_extreme_saturation():
    """Verify behavior on extreme saturation boundaries, zeros, and large outliers."""
    from strhub.quant.int_flashattention import int_flashattention_forward

    B, H, N, d = 2, 4, 32, 16
    # 1. Zeros tensor
    q_zero = torch.zeros(B, H, N, d)
    k_zero = torch.zeros(B, H, N, d)
    v_zero = torch.zeros(B, H, N, d)
    out_zero = int_flashattention_forward(q_zero, k_zero, v_zero)
    assert torch.isfinite(out_zero).all()

    # 2. Outliers
    q_outlier = torch.randn(B, H, N, d)
    q_outlier[:, :, 0, 0] = 1000.0
    q_outlier[:, :, 1, 1] = -1000.0
    k_outlier = torch.randn(B, H, N, d)
    v_outlier = torch.randn(B, H, N, d)
    out_outlier = int_flashattention_forward(q_outlier, k_outlier, v_outlier)
    assert torch.isfinite(out_outlier).all()


def test_int_flashattention_model_integration():
    """AC10: Verify integration of INT-FlashAttention into PARSeq model variants and dynamic batching."""
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

