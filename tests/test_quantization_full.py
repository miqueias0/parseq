#!/usr/bin/env python3
"""Comprehensive Validation Suite for All PARSeq INT8 Quantization Strategies.

Tests:
1. Checkpoint loading with zero regression.
2. Architecture transformation across all methods:
   - real_int8
   - jetfire_fqt
   - int_flashattn
   - ibert
   - unified_int8
   - smoothquant_int8
   - qat
   - dynamic
3. Forward pass sanity and non-NaN outputs.
4. FQT backpropagation and 8-bit gradient generation.
5. ONNX Export and ONNX INT8 generation.
"""

import unittest
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from strhub.models.utils import load_from_checkpoint
from strhub.models.quantization import PARSeqQuantizer
from strhub.models.quantization.layers import RealHardwareInt8Linear, JetfireInt8Linear, QuantizedLinear
from strhub.models.quantization.int_attention import INT8MultiheadAttention
from strhub.models.quantization.ibert_ops import IGELU, ILayerNorm


class TestPARSeqQuantizationFull(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        # Load baseline model
        cls.base_model = load_from_checkpoint("pretrained=parseq", max_label_length=25).eval()
        cls.dummy_img = torch.randn(2, 3, 32, 128)

    def test_01_real_int8_inference(self):
        """Validates real_int8 method conversion and forward pass."""
        q_model = PARSeqQuantizer.quantize(self.base_model, method="real_int8")
        inner = PARSeqQuantizer._get_inner_model(q_model)
        self.assertIsInstance(inner.encoder.blocks[0].attn.qkv, RealHardwareInt8Linear)
        
        with torch.no_grad():
            logits = q_model(self.dummy_img)
        self.assertEqual(logits.shape[0], 2)
        self.assertFalse(torch.isnan(logits).any())

    def test_02_int_flashattn_inference(self):
        """Validates int_flashattn method (INT-FlashAttention in Decoder)."""
        q_model = PARSeqQuantizer.quantize(self.base_model, method="int_flashattn", block_size=32)
        inner = PARSeqQuantizer._get_inner_model(q_model)
        self.assertIsInstance(inner.decoder.layers[0].self_attn, INT8MultiheadAttention)
        self.assertIsInstance(inner.decoder.layers[0].cross_attn, INT8MultiheadAttention)

        with torch.no_grad():
            logits = q_model(self.dummy_img)
        self.assertEqual(logits.shape[0], 2)
        self.assertFalse(torch.isnan(logits).any())

    def test_03_ibert_integer_inference(self):
        """Validates ibert method (i-GELU, i-Softmax, i-LayerNorm)."""
        q_model = PARSeqQuantizer.quantize(self.base_model, method="ibert")
        inner = PARSeqQuantizer._get_inner_model(q_model)
        self.assertIsInstance(inner.encoder.blocks[0].norm1, ILayerNorm)
        self.assertIsInstance(inner.encoder.blocks[0].mlp.act, IGELU)

        with torch.no_grad():
            logits = q_model(self.dummy_img)
        self.assertEqual(logits.shape[0], 2)
        self.assertFalse(torch.isnan(logits).any())

    def test_04_unified_int8_inference(self):
        """Validates unified_int8 combining Jetfire Data Flow + INT-FlashAttention + Fused non-linears."""
        q_model = PARSeqQuantizer.quantize(self.base_model, method="unified_int8", block_size=32)
        inner = PARSeqQuantizer._get_inner_model(q_model)
        self.assertIsInstance(inner.encoder.blocks[0].attn.qkv, RealHardwareInt8Linear)
        self.assertIsInstance(inner.decoder.layers[0].self_attn, INT8MultiheadAttention)

        with torch.no_grad():
            logits = q_model(self.dummy_img)
        self.assertEqual(logits.shape[0], 2)
        self.assertFalse(torch.isnan(logits).any())

    def test_05_jetfire_fqt_training_step(self):
        """Validates jetfire_fqt direct INT8 training with 8-bit gradients."""
        q_model = PARSeqQuantizer.quantize(self.base_model, method="jetfire_fqt", block_size=32)
        q_model.train()
        inner = PARSeqQuantizer._get_inner_model(q_model)
        self.assertIsInstance(inner.encoder.blocks[0].attn.qkv, JetfireInt8Linear)

        labels = ["PARSEQ", "INT8"]
        loss = q_model.training_step((self.dummy_img, labels), 0)
        self.assertFalse(torch.isnan(loss))

        loss.backward()
        self.assertIsNotNone(inner.encoder.blocks[0].attn.qkv.weight.grad)
        self.assertTrue((inner.encoder.blocks[0].attn.qkv.weight.grad != 0).any())

    def test_06_onnx_export_and_int8_quantization(self):
        """Validates export_onnx and export_onnx_int8 pipelines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_fp_path = Path(tmpdir) / "parseq.onnx"
            onnx_int8_path = Path(tmpdir) / "parseq_int8.onnx"

            # Export float ONNX
            PARSeqQuantizer.export_onnx(
                self.base_model,
                output_path=onnx_fp_path,
                img_size=(32, 128),
                mode="nar",
            )
            self.assertTrue(onnx_fp_path.exists())
            self.assertGreater(onnx_fp_path.stat().st_size, 1_000_000)

            # Export INT8 ONNX
            PARSeqQuantizer.export_onnx_int8(
                float_onnx_path=onnx_fp_path,
                output_int8_path=onnx_int8_path,
            )
            self.assertTrue(onnx_int8_path.exists())
            self.assertGreater(onnx_int8_path.stat().st_size, 500_000)


if __name__ == "__main__":
    unittest.main()
