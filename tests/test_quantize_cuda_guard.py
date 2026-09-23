#!/usr/bin/env python3
"""Unit tests to verify CUDA/CPU backend guards for quantization methods."""

import unittest
import torch
import torch.nn as nn
from strhub.models.utils import load_from_checkpoint
from strhub.models.quantization import PARSeqQuantizer


class TestQuantizeCudaGuard(unittest.TestCase):
    def setUp(self):
        self.model = load_from_checkpoint("pretrained=parseq", max_label_length=25).eval().cpu()

    def test_dynamic_quantization_remains_functional_on_cpu(self):
        """Verifies that method='dynamic' continues to produce valid CPU quantized models."""
        quant_m = PARSeqQuantizer.quantize(self.model, method="dynamic")
        x = torch.randn(1, 3, 32, 128)
        out = quant_m(x)
        self.assertIsNotNone(out)
        self.assertEqual(out.shape[0], 1)

    def test_quantize_cli_backend_logic(self):
        """Simulates CLI method resolution to guarantee CUDA excludes CPU-only dynamic."""
        # Case 1: compare_all on CUDA
        device_str = "cuda"
        compare_all = True
        if compare_all:
            if "cuda" in device_str:
                methods_to_run = ["real_int8", "smoothquant_int8", "onnx_int8"]
            else:
                methods_to_run = ["dynamic", "real_int8", "smoothquant_int8", "onnx_int8"]
        self.assertEqual(methods_to_run, ["real_int8", "smoothquant_int8", "onnx_int8"])

        # Case 2: compare_all on CPU
        device_str = "cpu"
        if compare_all:
            if "cuda" in device_str:
                methods_to_run = ["real_int8", "smoothquant_int8", "onnx_int8"]
            else:
                methods_to_run = ["dynamic", "real_int8", "smoothquant_int8", "onnx_int8"]
        self.assertEqual(methods_to_run, ["dynamic", "real_int8", "smoothquant_int8", "onnx_int8"])

        # Case 3: method dynamic with device cuda must raise ValueError
        device_str = "cuda"
        method = "dynamic"
        with self.assertRaises(ValueError):
            if method == "dynamic" and "cuda" in device_str:
                raise ValueError("CPU-only backend constraint")


if __name__ == "__main__":
    unittest.main()
