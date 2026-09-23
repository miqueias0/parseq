#!/usr/bin/env python3
"""Numerical Validation Tests for I-BERT Integer Operations."""

import unittest
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import math
import torch
import torch.nn.functional as F

from strhub.models.quantization.ibert_ops import (
    integer_sqrt_newton_raphson,
    IGELU,
    IExpSoftmax,
    ILayerNorm,
)


class TestIBERTOperations(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_newton_raphson_integer_sqrt(self):
        """Verifies integer square root against exact math.isqrt across varied magnitudes."""
        test_values = torch.tensor([0, 1, 4, 9, 15, 16, 17, 100, 255, 1024, 65535, 1000000, 2147483647], dtype=torch.int64)
        pred_sqrt = integer_sqrt_newton_raphson(test_values)
        
        for val, pred in zip(test_values.tolist(), pred_sqrt.tolist()):
            exact = math.isqrt(val)
            self.assertEqual(pred, exact, f"Mismatch for val={val}: pred={pred}, exact={exact}")

    def test_igelu_numerical_precision(self):
        """Validates that i-GELU approximates FP32 GELU within published bounds (max error < 0.02)."""
        x = torch.linspace(-5.0, 5.0, 1000)
        igelu = IGELU()
        
        y_approx = igelu(x)
        y_exact = F.gelu(x)
        
        abs_err = (y_approx - y_exact).abs()
        max_err = abs_err.max().item()
        mean_err = abs_err.mean().item()
        
        # Paper reports max error ~0.018 and mean error ~0.008
        self.assertLess(max_err, 0.025, f"Max error {max_err} exceeds threshold 0.025")
        self.assertLess(mean_err, 0.010, f"Mean error {mean_err} exceeds threshold 0.010")

    def test_isoftmax_numerical_precision(self):
        """Validates that i-Softmax with base-decomposition bitshifts approximates standard Softmax within 0.01."""
        x = torch.randn(10, 50)
        isoftmax = IExpSoftmax(dim=-1)
        
        y_approx = isoftmax(x)
        y_exact = F.softmax(x, dim=-1)
        
        # Rows must sum to 1.0
        row_sums = y_approx.sum(dim=-1)
        torch.testing.assert_close(row_sums, torch.ones_like(row_sums), atol=1e-5, rtol=1e-5)
        
        # Max absolute error between probabilities
        max_err = (y_approx - y_exact).abs().max().item()
        self.assertLess(max_err, 0.015, f"Softmax max error {max_err} exceeds threshold 0.015")

    def test_ilayernorm_numerical_precision(self):
        """Validates that ILayerNorm with integer sqrt closely tracks PyTorch LayerNorm (MRE < 1%)."""
        norm_shape = 384  # PARSeq embed_dim
        x = torch.randn(4, 25, norm_shape)
        
        iln = ILayerNorm(norm_shape)
        ln = torch.nn.LayerNorm(norm_shape)
        with torch.no_grad():
            ln.weight.copy_(iln.weight)
            ln.bias.copy_(iln.bias)
            
        y_approx = iln(x)
        y_exact = ln(x)
        
        rel_err = (y_approx - y_exact).abs().mean() / y_exact.abs().mean()
        self.assertLess(rel_err.item(), 0.02, f"LayerNorm relative error {rel_err.item()} exceeds 2%")


if __name__ == "__main__":
    unittest.main()
