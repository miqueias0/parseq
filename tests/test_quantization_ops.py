"""
Unit Tests for PARSeq Integer-Only and INT8 Quantization Framework
==================================================================
Tests:
- Dyadic arithmetic and integer bit length
- Newton-Raphson integer square root (I-BERT)
- Observers (MinMax, PerChannel, KL Histogram)
- Quant-Noise stochastic layers (Fan et al., 2021)
- DataAwarePolyGELU numerical accuracy and integer mode (IPTQ-ViT)
- EfficientBitSoftmax numerical accuracy and integer mode (IPTQ-ViT)
- IntegerLayerNorm and DyadicResidualAdd (HAWQ-V3)
- QuantizedPatchEmbed end-to-end integer forward
- Unified Metric (Omega) search routine
- Full IntegerPARSeq model execution
"""

import math
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

from strhub.models.utils import create_model
from strhub.quantization.core import (
    DyadicResidualAdd,
    EfficientBitSoftmax,
    DataAwarePolyGELU,
    IntegerLayerNorm,
    IntegerLinear,
    IntegerMatMul,
    KLHistogramObserver,
    MinMaxObserver,
    PerChannelMinMaxObserver,
    QuantNoiseConv2d,
    QuantNoiseLinear,
    QuantizedPatchEmbed,
    dyadic_scale,
    float_to_dyadic,
    integer_sqrt_newton,
    pure_int_bit_length,
)
from strhub.quantization.parseq_quantizer import PARSeqQuantizer, quantize_parseq
from strhub.quantization.unified_metric import (
    UnifiedMetricCalculator,
    UnifiedMetricSearcher,
    compute_perturbation,
    compute_sqnr,
    compute_unified_metric,
)


class TestQuantizationOps(unittest.TestCase):

    def test_dyadic_arithmetic(self):
        """Tests float to dyadic conversion and dyadic scaling."""
        scales = [0.05, 0.125, 0.3333, 1.0, 2.5]
        for s in scales:
            b, c = float_to_dyadic(s)
            approx = b / (1 << c)
            self.assertAlmostEqual(s, approx, places=3, msg=f"Dyadic approximation failed for scale {s}")

        # Test dyadic scaling on integer tensor
        x = torch.tensor([0, 10, 50, 100, -20], dtype=torch.int64)
        b, c = float_to_dyadic(0.5)
        res = dyadic_scale(x, b, c)
        expected = torch.tensor([0, 5, 25, 50, -10], dtype=torch.int64)
        self.assertTrue(torch.equal(res, expected))

    def test_pure_int_bit_length(self):
        """Tests pure integer bit length calculation without floating point."""
        vals = torch.tensor([1, 2, 3, 4, 7, 8, 15, 16, 100, 10000, 2147483647], dtype=torch.int64)
        bl = pure_int_bit_length(vals)
        expected = torch.tensor([v.bit_length() for v in vals.tolist()], dtype=torch.int64)
        self.assertTrue(torch.equal(bl, expected))

    def test_newton_integer_sqrt(self):
        """Tests I-BERT Newton-Raphson integer square root."""
        test_vals = [0, 1, 2, 3, 4, 8, 9, 15, 16, 25, 99, 100, 1000, 65536, 2147483647]
        t_vals = torch.tensor(test_vals, dtype=torch.int64)
        res = integer_sqrt_newton(t_vals)
        expected = torch.tensor([int(math.isqrt(v)) for v in test_vals], dtype=torch.int64)
        self.assertTrue(torch.equal(res, expected), f"Mismatch: res={res.tolist()}, expected={expected.tolist()}")

    def test_observers(self):
        """Tests MinMax, PerChannel, and KL Histogram observers."""
        x = torch.randn(4, 16, 8, 8) * 2.0
        # MinMax
        minmax = MinMaxObserver(bits=8)
        minmax(x)
        minmax.finish_calibration()
        self.assertGreater(minmax.scale.item(), 0.0)

        # PerChannel
        per_ch = PerChannelMinMaxObserver(ch_axis=1, bits=8)
        per_ch(x)
        self.assertEqual(per_ch.scale.shape[0], 16)

        # KL Histogram
        kl = KLHistogramObserver(num_bins=256, bits=8)
        kl(x)
        kl.collect_histogram(x)
        kl.finish_calibration()
        self.assertGreater(kl.scale.item(), 0.0)

    def test_quant_noise_layers(self):
        """Tests stochastic Quant-Noise layers."""
        linear = QuantNoiseLinear(16, 32, p=0.5, bits=8)
        x = torch.randn(2, 16, requires_grad=True)

        # Train mode: stochastic noise applied
        linear.train()
        out = linear(x)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(linear.weight.grad)

        # Eval mode: full quantization applied
        linear.eval()
        out_eval = linear(x)
        self.assertEqual(out_eval.shape, (2, 32))

    def test_data_aware_poly_gelu(self):
        """Tests IPTQ-ViT Data-aware Poly-GELU error against FP32 GELU."""
        x = torch.linspace(-4.0, 4.0, 200)
        poly_gelu = DataAwarePolyGELU(scale_in=0.05, scale_out=0.05, integer_only=False)
        out = poly_gelu(x)
        true_gelu = F.gelu(x)

        l2_err = torch.sqrt(torch.mean((out - true_gelu) ** 2)).item()
        max_err = torch.max(torch.abs(out - true_gelu)).item()
        # IPTQ-ViT paper reports L2 error 0.0051 and L_inf error 0.0093
        self.assertLess(l2_err, 0.015, f"L2 error too high: {l2_err}")
        self.assertLess(max_err, 0.02, f"Max error too high: {max_err}")

        # Integer-only mode execution
        poly_gelu_int = DataAwarePolyGELU(scale_in=0.05, scale_out=0.05, integer_only=True)
        q_x = torch.randint(-128, 127, (100,), dtype=torch.int32)
        out_int = poly_gelu_int(q_x)
        self.assertEqual(out_int.dtype, torch.int8)

    def test_efficient_bit_softmax(self):
        """Tests IPTQ-ViT Efficient Bit-Softmax against FP32 Softmax."""
        logits = torch.randn(2, 4, 16)
        scale = 0.05
        q_logits = torch.round(logits / scale).to(torch.int32)

        bit_softmax = EfficientBitSoftmax(scale_in=scale, bit_width=8, M=28)
        q_out = bit_softmax(q_logits)

        reconstructed = q_out.float() / 128.0
        fp_softmax = F.softmax(logits, dim=-1)
        l1_err = torch.mean(torch.abs(reconstructed - fp_softmax)).item()
        self.assertLess(l1_err, 0.025, f"L1 error too high: {l1_err}")
        self.assertEqual(q_out.dtype, torch.int8)

    def test_integer_layernorm(self):
        """Tests I-BERT IntegerLayerNorm with Newton integer sqrt."""
        dim = 64
        int_ln = IntegerLayerNorm(normalized_shape=dim, scale_in=0.05, scale_out=0.05)
        fp_w = torch.ones(dim)
        fp_b = torch.zeros(dim)
        int_ln.set_parameters(fp_w, fp_b, scale_in=0.05, scale_out=0.05)

        q_x = torch.randint(-60, 60, (2, 10, dim), dtype=torch.int32)
        out = int_ln(q_x)
        self.assertEqual(out.dtype, torch.int8)
        self.assertTrue((out >= -128).all() and (out <= 127).all())

    def test_dyadic_residual_add(self):
        """Tests DyadicResidualAdd."""
        res_add = DyadicResidualAdd(scale_main=0.05, scale_res=0.05, scale_out=0.05)
        q_m = torch.tensor([10, 20, 30], dtype=torch.int32)
        q_r = torch.tensor([5, -10, 20], dtype=torch.int32)
        out = res_add(q_m, q_r)
        self.assertTrue(torch.equal(out, torch.tensor([15, 10, 50], dtype=torch.int8)))

    def test_quantized_patch_embed(self):
        """Tests QuantizedPatchEmbed with integer convolution and positional embedding addition."""
        patch_embed = QuantizedPatchEmbed(
            img_size=(32, 128),
            patch_size=(4, 4),
            in_chans=3,
            embed_dim=128,
        )
        fp_w = torch.randn(128, 3, 4, 4)
        fp_b = torch.randn(128)
        num_patches = (32 // 4) * (128 // 4)
        fp_pos = torch.randn(1, num_patches, 128)
        patch_embed.set_quantized_parameters(fp_w, fp_b, fp_pos, 1.0/127.0, 0.05, 0.05, 0.05)

        img = torch.randint(-128, 127, (2, 3, 32, 128), dtype=torch.int8)
        tokens = patch_embed(img)
        self.assertEqual(tokens.shape, (2, num_patches, 128))
        self.assertEqual(tokens.dtype, torch.int8)

    def test_unified_metric_searcher(self):
        """Tests IPTQ-ViT Unified Metric search engine."""
        searcher = UnifiedMetricSearcher()
        x = torch.randn(4, 16, 64)
        best_gelu = searcher.search_layer("block0.gelu", "gelu", x)
        best_ln = searcher.search_layer("block0.ln", "layernorm", x)
        best_sm = searcher.search_layer("block0.softmax", "softmax", torch.randn(4, 4, 16, 16))

        self.assertIn(best_gelu, ["DataAwarePolyGELU", "i-GELU", "BitShiftGELU"])
        self.assertIn(best_ln, ["IntegerLayerNorm", "BitShiftLayerNorm"])
        self.assertIn(best_sm, ["EfficientBitSoftmax", "Shiftmax", "LogSoftmax"])

    def test_end_to_end_parseq_quantization(self):
        """Tests full PARSeq conversion to integer_only mode."""
        base_model = create_model("parseq", pretrained=False)
        base_model.eval()

        # Build pure integer PARSeq model
        int_model = quantize_parseq(base_model, mode="integer_only")
        int_img = torch.randint(-128, 127, (2, 3, 32, 128), dtype=torch.int8)

        # Forward pass in integer arithmetic
        logits = int_model(int_img, max_length=5)
        head = getattr(base_model, "head", getattr(base_model, "model", None).head)
        self.assertEqual(logits.shape[2], head.out_features)


if __name__ == "__main__":
    unittest.main()
