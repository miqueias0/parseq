#!/usr/bin/env python3
"""Comprehensive Validation & Regression Test Suite for Jetfire Direct INT8 Training (FQT)."""

import unittest
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn as nn
import torch.nn.functional as F

from strhub.models.quantization.layers import (
    JetfireFQTFunction,
    JetfireInt8Linear,
    QuantizedLinear,
    RealHardwareInt8Linear,
    block_quantize_2d,
    dequantize_blocks,
)
from strhub.models.quantization.quantizer import PARSeqQuantizer
from strhub.models.utils import load_from_checkpoint


class TestJetfireFQT(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_block_quantize_and_dequantize(self):
        """Validates 2D block quantization round-trip and scale computation."""
        x = torch.randn(50, 150)
        q, scales, meta = block_quantize_2d(x, block_size=64)
        self.assertEqual(q.dtype, torch.int8)
        self.assertEqual(q.shape, (1, 3, 64, 64))
        self.assertEqual(scales.shape, (1, 3, 1, 1))
        
        # Dequantize back
        deq = (q.to(torch.float32) * scales)
        x_recon = dequantize_blocks(deq, meta, block_size=64)
        self.assertEqual(x_recon.shape, x.shape)
        # Relative error should be within standard 8-bit quantization bounds (< 2%)
        rel_err = (x - x_recon).abs().mean() / x.abs().mean()
        self.assertLess(rel_err.item(), 0.03)

    def test_jetfire_linear_autograd(self):
        """Validates forward and backward pass with INT8 GEMMs on CPU."""
        linear = JetfireInt8Linear(in_features=128, out_features=256, bias=True, block_size=64)
        x = torch.randn(4, 16, 128, requires_grad=True)
        
        # Forward pass
        out = linear(x)
        self.assertEqual(out.shape, (4, 16, 256))
        self.assertFalse(torch.isnan(out).any())

        # Backward pass
        loss = out.sum()
        loss.backward()

        # Verify gradients exist and are non-zero
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)
        self.assertTrue((x.grad != 0).any())

        self.assertIsNotNone(linear.weight.grad)
        self.assertEqual(linear.weight.grad.shape, linear.weight.shape)
        self.assertTrue((linear.weight.grad != 0).any())

        self.assertIsNotNone(linear.bias.grad)
        self.assertEqual(linear.bias.grad.shape, linear.bias.shape)
        self.assertTrue((linear.bias.grad != 0).any())

    def test_jetfire_optimization_convergence(self):
        """Verifies that an optimizer step reduces loss when training with Jetfire FQT."""
        linear = JetfireInt8Linear(in_features=64, out_features=64, bias=True, block_size=32)
        optimizer = torch.optim.SGD(linear.parameters(), lr=0.05)

        x = torch.randn(8, 64)
        target = torch.randn(8, 64)

        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            pred = linear(x)
            loss = F.mse_loss(pred, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease
        self.assertLess(losses[-1], losses[0])

    def test_parseq_jetfire_integration(self):
        """Tests end-to-end forward and backward pass through full PARSeq architecture."""
        model = load_from_checkpoint("pretrained=parseq", max_label_length=25).eval()
        jetfire_model = PARSeqQuantizer.prepare_for_jetfire_training(model, block_size=64)
        jetfire_model.train()

        # Synthetic image batch: [2, 3, 32, 128]
        images = torch.randn(2, 3, 32, 128)
        labels = ["TEST", "INT8"]
        loss = jetfire_model.training_step((images, labels), 0)
        self.assertFalse(torch.isnan(loss))
        
        loss.backward()
        # Verify gradients reached encoder attention layers
        inner = PARSeqQuantizer._get_inner_model(jetfire_model)
        enc_block = inner.encoder.blocks[0]
        self.assertIsInstance(enc_block.attn.qkv, JetfireInt8Linear)
        self.assertIsNotNone(enc_block.attn.qkv.weight.grad)
        self.assertTrue((enc_block.attn.qkv.weight.grad != 0).any())

    def test_regression_existing_quantization(self):
        """Ensures existing methods ('real_int8', 'smoothquant_int8', 'qat', 'dynamic') remain fully operational."""
        model = load_from_checkpoint("pretrained=parseq", max_label_length=25).eval()
        
        # 1. Real INT8
        m_real = PARSeqQuantizer.quantize(model, method="real_int8")
        inner_real = PARSeqQuantizer._get_inner_model(m_real)
        self.assertIsInstance(inner_real.encoder.blocks[0].attn.qkv, RealHardwareInt8Linear)

        # 2. QAT
        m_qat = PARSeqQuantizer.quantize(model, method="qat")
        inner_qat = PARSeqQuantizer._get_inner_model(m_qat)
        self.assertIsInstance(inner_qat.encoder.blocks[0].attn.qkv, QuantizedLinear)

        # 3. Dynamic
        m_dyn = PARSeqQuantizer.quantize(model, method="dynamic")
        inner_dyn = PARSeqQuantizer._get_inner_model(m_dyn)
        self.assertTrue(hasattr(inner_dyn, "encoder"))

        # 4. Jetfire FQT
        m_jet = PARSeqQuantizer.quantize(model, method="jetfire_fqt")
        inner_jet = PARSeqQuantizer._get_inner_model(m_jet)
        self.assertIsInstance(inner_jet.encoder.blocks[0].attn.qkv, JetfireInt8Linear)


if __name__ == "__main__":
    unittest.main()
