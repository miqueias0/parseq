#!/usr/bin/env python3
"""Comprehensive Pipeline Runner for all PARSeq Architecture and Quantization Variants:
Runs:
1. Automated unit & scientific validation tests.
2. ONNX Export across all variants (M0..M6, fusion levels, INT-FlashAttention).
3. TensorRT Engine Compilation across all precision modes (FP32, FP16, INT8).
4. TensorRT Latency and Throughput Benchmarking (Batch 1 and Batch 32).
5. ALPR Accuracy Validation (Exact Plate Accuracy, NED, CER).
6. Generation of consolidated scientific reports.
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, Any, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import torch
from tools.export_onnx import export_onnx
from tools.build_tensorrt import build_tensorrt_engine
from tools.benchmark_tensorrt import benchmark_tensorrt
from tools.evaluate_alpr import TensorRTModelWrapper, evaluate_dataset
from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint


CONFIGURATIONS = [
    # =========================================================================
    # 1. Floating-Point Baselines & Controls
    # =========================================================================
    {
        "id": "m0_ar_fp32",
        "label": "M0: FP32 AR Baseline (1 iter)",
        "variant": "m0",
        "ckpt": "pretrained/parseq_alpr_98.5.ckpt",
        "precision": "fp32",
        "fusion_level": "none",
        "use_int_fa": False,
        "onnx_path": "onnx/parseq_m0_ar_fp32.onnx",
        "engine_path": "trt/parseq_m0_ar_fp32.engine",
    },
    {
        "id": "m1_nar_fp32",
        "label": "M1: FP32 NAR Baseline",
        "variant": "m1",
        "ckpt": "pretrained/parseq_alpr_98.5.ckpt",
        "precision": "fp32",
        "fusion_level": "none",
        "use_int_fa": False,
        "onnx_path": "onnx/parseq_m1_nar_fp32.onnx",
        "engine_path": "trt/parseq_m1_nar_fp32.engine",
    },
    {
        "id": "m2_nar_fp16",
        "label": "M2: FP16 NAR Baseline",
        "variant": "m2",
        "ckpt": "pretrained/parseq_alpr_98.5.ckpt",
        "precision": "fp16",
        "fusion_level": "none",
        "use_int_fa": False,
        "onnx_path": "onnx/parseq_m2_nar_fp16.onnx",
        "engine_path": "trt/parseq_m2_nar_fp16.engine",
    },
    {
        "id": "m3_nar_int8_naive",
        "label": "M3: Naive INT8 NAR (Negative Control)",
        "variant": "m3",
        "ckpt": "pretrained/parseq_alpr_98.5.ckpt",
        "precision": "int8",
        "fusion_level": "none",
        "use_int_fa": False,
        "onnx_path": "onnx/parseq_m3_nar_int8_naive.onnx",
        "engine_path": "trt/parseq_m3_nar_int8_naive.engine",
    },
    {
        "id": "m4_nar_int8_ptq",
        "label": "M4: Conventional W8A8 PTQ Baseline",
        "variant": "m4",
        "ckpt": "pretrained/parseq_alpr_98.5.ckpt",
        "precision": "int8",
        "fusion_level": "none",
        "use_int_fa": False,
        "onnx_path": "onnx/parseq_m4_nar_int8_ptq.onnx",
        "engine_path": "trt/parseq_m4_nar_int8_ptq.engine",
    },
    {
        "id": "m4_fused_all",
        "label": "M4: Conventional W8A8 PTQ (All Fused)",
        "variant": "m4",
        "ckpt": "pretrained/parseq_alpr_98.5.ckpt",
        "precision": "int8",
        "fusion_level": "all",
        "use_int_fa": False,
        "onnx_path": "onnx/parseq_m4_fused_all.onnx",
        "engine_path": "trt/parseq_m4_fused_all.engine",
    },

    # =========================================================================
    # 2. M5: Integer-Only PTQ - Progressive Kernel Fusion (none -> shapes -> mha -> mlp -> all)
    # =========================================================================
    {
        "id": "m5_nar_int8_io_ptq",
        "label": "M5: Integer-Only PTQ Baseline",
        "variant": "m5",
        "ckpt": "pretrained/parseq_alpr_98.5.ckpt",
        "precision": "int8",
        "fusion_level": "none",
        "use_int_fa": False,
        "onnx_path": "onnx/parseq_m5_nar_int8_io_ptq.onnx",
        "engine_path": "trt/parseq_m5_nar_int8_io_ptq.engine",
    },
    # {
    #     "id": "m5_fused_shapes",
    #     "label": "M5: Integer-Only PTQ + Shapes",
    #     "variant": "m5",
    #     "ckpt": "pretrained/parseq_alpr_98.5.ckpt",
    #     "precision": "int8",
    #     "fusion_level": "shapes",
    #     "use_int_fa": False,
    #     "onnx_path": "onnx/parseq_m5_fused_shapes.onnx",
    #     "engine_path": "trt/parseq_m5_fused_shapes.engine",
    # },
    # {
    #     "id": "m5_fused_mha",
    #     "label": "M5: Integer-Only PTQ + Shapes + MHA",
    #     "variant": "m5",
    #     "ckpt": "pretrained/parseq_alpr_98.5.ckpt",
    #     "precision": "int8",
    #     "fusion_level": "mha",
    #     "use_int_fa": False,
    #     "onnx_path": "onnx/parseq_m5_fused_mha.onnx",
    #     "engine_path": "trt/parseq_m5_fused_mha.engine",
    # },
    # {
    #     "id": "m5_fused_mlp",
    #     "label": "M5: Integer-Only PTQ + Shapes + MHA + MLP",
    #     "variant": "m5",
    #     "ckpt": "pretrained/parseq_alpr_98.5.ckpt",
    #     "precision": "int8",
    #     "fusion_level": "mlp",
    #     "use_int_fa": False,
    #     "onnx_path": "onnx/parseq_m5_fused_mlp.onnx",
    #     "engine_path": "trt/parseq_m5_fused_mlp.engine",
    # },
    {
        "id": "m5_fused_all",
        "label": "M5: Integer-Only PTQ (All Kernels Fused)",
        "variant": "m5",
        "ckpt": "pretrained/parseq_alpr_98.5.ckpt",
        "precision": "int8",
        "fusion_level": "all",
        "use_int_fa": False,
        "onnx_path": "onnx/parseq_m5_fused_all.onnx",
        "engine_path": "trt/parseq_m5_fused_all.engine",
    },

    # =========================================================================
    # 3. M6: Integer-Only QAT - Progressive Kernel Fusion (none -> shapes -> mha -> mlp -> all)
    # =========================================================================
    {
        "id": "m6_nar_int8_io_qat",
        "label": "M6: Integer-Only QAT Baseline",
        "variant": "m6",
        "ckpt": "pretrained/parseq_alpr_qat_m6.ckpt",
        "precision": "int8",
        "fusion_level": "none",
        "use_int_fa": False,
        "onnx_path": "onnx/parseq_m6_nar_int8_io_qat.onnx",
        "engine_path": "trt/parseq_m6_nar_int8_io_qat.engine",
    },
    # {
    #     "id": "m6_fused_shapes",
    #     "label": "M6: Integer-Only QAT + Shapes",
    #     "variant": "m6",
    #     "ckpt": "pretrained/parseq_alpr_qat_m6.ckpt",
    #     "precision": "int8",
    #     "fusion_level": "shapes",
    #     "use_int_fa": False,
    #     "onnx_path": "onnx/parseq_m6_fused_shapes.onnx",
    #     "engine_path": "trt/parseq_m6_fused_shapes.engine",
    # },
    # {
    #     "id": "m6_fused_mha",
    #     "label": "M6: Integer-Only QAT + Shapes + MHA",
    #     "variant": "m6",
    #     "ckpt": "pretrained/parseq_alpr_qat_m6.ckpt",
    #     "precision": "int8",
    #     "fusion_level": "mha",
    #     "use_int_fa": False,
    #     "onnx_path": "onnx/parseq_m6_fused_mha.onnx",
    #     "engine_path": "trt/parseq_m6_fused_mha.engine",
    # },
    # {
    #     "id": "m6_fused_mlp",
    #     "label": "M6: Integer-Only QAT + Shapes + MHA + MLP",
    #     "variant": "m6",
    #     "ckpt": "pretrained/parseq_alpr_qat_m6.ckpt",
    #     "precision": "int8",
    #     "fusion_level": "mlp",
    #     "use_int_fa": False,
    #     "onnx_path": "onnx/parseq_m6_fused_mlp.onnx",
    #     "engine_path": "trt/parseq_m6_fused_mlp.engine",
    # },
    {
        "id": "m6_fused_all",
        "label": "M6: Integer-Only QAT (All Kernels Fused)",
        "variant": "m6",
        "ckpt": "pretrained/parseq_alpr_qat_m6.ckpt",
        "precision": "int8",
        "fusion_level": "all",
        "use_int_fa": False,
        "onnx_path": "onnx/parseq_m6_fused_all.onnx",
        "engine_path": "trt/parseq_m6_fused_all.engine",
    },

    # =========================================================================
    # 4. M5: Integer-Only PTQ + INT-FlashAttention (Progressive Fusion)
    # =========================================================================
    # {
    #     "id": "m5_int_fa",
    #     "label": "M5: Integer-Only PTQ + INT-FlashAttention",
    #     "variant": "m5",
    #     "ckpt": "pretrained/parseq_alpr_98.5.ckpt",
    #     "precision": "int8",
    #     "fusion_level": "none",
    #     "use_int_fa": True,
    #     "onnx_path": "onnx/parseq_m5_int_fa.onnx",
    #     "engine_path": "trt/parseq_m5_int_fa.engine",
    # },
    # {
    #     "id": "m5_int_fa_fuse_shapes",
    #     "label": "M5: Integer-Only PTQ + INT-Flash + Shapes",
    #     "variant": "m5",
    #     "ckpt": "pretrained/parseq_alpr_98.5.ckpt",
    #     "precision": "int8",
    #     "fusion_level": "shapes",
    #     "use_int_fa": True,
    #     "onnx_path": "onnx/parseq_m5_int_fa_fuse_shapes.onnx",
    #     "engine_path": "trt/parseq_m5_int_fa_fuse_shapes.engine",
    # },
    # {
    #     "id": "m5_int_fa_fuse_mha",
    #     "label": "M5: Integer-Only PTQ + INT-Flash + Shapes + MHA",
    #     "variant": "m5",
    #     "ckpt": "pretrained/parseq_alpr_98.5.ckpt",
    #     "precision": "int8",
    #     "fusion_level": "mha",
    #     "use_int_fa": True,
    #     "onnx_path": "onnx/parseq_m5_int_fa_fuse_mha.onnx",
    #     "engine_path": "trt/parseq_m5_int_fa_fuse_mha.engine",
    # },
    # {
    #     "id": "m5_int_fa_fuse_mlp",
    #     "label": "M5: Integer-Only PTQ + INT-Flash + Shapes + MHA + MLP",
    #     "variant": "m5",
    #     "ckpt": "pretrained/parseq_alpr_98.5.ckpt",
    #     "precision": "int8",
    #     "fusion_level": "mlp",
    #     "use_int_fa": True,
    #     "onnx_path": "onnx/parseq_m5_int_fa_fuse_mlp.onnx",
    #     "engine_path": "trt/parseq_m5_int_fa_fuse_mlp.engine",
    # },
    # {
    #     "id": "m5_int_fa_fuse_all",
    #     "label": "M5: Integer-Only PTQ + INT-Flash + All Fused",
    #     "variant": "m5",
    #     "ckpt": "pretrained/parseq_alpr_98.5.ckpt",
    #     "precision": "int8",
    #     "fusion_level": "all",
    #     "use_int_fa": True,
    #     "onnx_path": "onnx/parseq_m5_int_fa_fuse_all.onnx",
    #     "engine_path": "trt/parseq_m5_int_fa_fuse_all_int8.engine",
    # },

    # =========================================================================
    # 5. M6: Integer-Only QAT + INT-FlashAttention (Progressive Fusion)
    # =========================================================================
    # {
    #     "id": "m6_int_fa",
    #     "label": "M6: Integer-Only QAT + INT-FlashAttention",
    #     "variant": "m6",
    #     "ckpt": "pretrained/parseq_alpr_qat_m6.ckpt",
    #     "precision": "int8",
    #     "fusion_level": "none",
    #     "use_int_fa": True,
    #     "onnx_path": "onnx/parseq_m6_int_fa.onnx",
    #     "engine_path": "trt/parseq_m6_int_fa.engine",
    # },
    # {
    #     "id": "m6_int_fa_fuse_shapes",
    #     "label": "M6: Integer-Only QAT + INT-Flash + Shapes",
    #     "variant": "m6",
    #     "ckpt": "pretrained/parseq_alpr_qat_m6.ckpt",
    #     "precision": "int8",
    #     "fusion_level": "shapes",
    #     "use_int_fa": True,
    #     "onnx_path": "onnx/parseq_m6_int_fa_fuse_shapes.onnx",
    #     "engine_path": "trt/parseq_m6_int_fa_fuse_shapes.engine",
    # },
    # {
    #     "id": "m6_int_fa_fuse_mha",
    #     "label": "M6: Integer-Only QAT + INT-Flash + Shapes + MHA",
    #     "variant": "m6",
    #     "ckpt": "pretrained/parseq_alpr_qat_m6.ckpt",
    #     "precision": "int8",
    #     "fusion_level": "mha",
    #     "use_int_fa": True,
    #     "onnx_path": "onnx/parseq_m6_int_fa_fuse_mha.onnx",
    #     "engine_path": "trt/parseq_m6_int_fa_fuse_mha.engine",
    # },
    # {
    #     "id": "m6_int_fa_fuse_mlp",
    #     "label": "M6: Integer-Only QAT + INT-Flash + Shapes + MHA + MLP",
    #     "variant": "m6",
    #     "ckpt": "pretrained/parseq_alpr_qat_m6.ckpt",
    #     "precision": "int8",
    #     "fusion_level": "mlp",
    #     "use_int_fa": True,
    #     "onnx_path": "onnx/parseq_m6_int_fa_fuse_mlp.onnx",
    #     "engine_path": "trt/parseq_m6_int_fa_fuse_mlp.engine",
    # },
    # {
    #     "id": "m6_int_fa_fuse_all",
    #     "label": "M6: Integer-Only QAT + INT-Flash + All Fused",
    #     "variant": "m6",
    #     "ckpt": "pretrained/parseq_alpr_qat_m6.ckpt",
    #     "precision": "int8",
    #     "fusion_level": "all",
    #     "use_int_fa": True,
    #     "onnx_path": "onnx/parseq_m6_int_fa_fuse_all.onnx",
    #     "engine_path": "trt/parseq_m6_int_fa_fuse_all.engine",
    # },

    # =========================================================================
    # 6. M5: SageAttention (Paper 2410.02367v9 - SAGEAttn-B & SAGEAttn-vB)
    # =========================================================================
    # Mode A: SAGEAttn-B (Float V) Progressive Fusion
    # {
    #     "id": "m5_sage_b",
    #     "label": "M5: Integer-Only PTQ + SageAttention (SAGEAttn-B)",
    #     "variant": "m5",
    #     "ckpt": "pretrained/parseq_alpr_98.5.ckpt",
    #     "precision": "int8",
    #     "fusion_level": "none",
    #     "use_int_fa": False,
    #     "use_sage": True,
    #     "sage_mode": "sageattn_b",
    #     "onnx_path": "onnx/parseq_m5_sage_b.onnx",
    #     "engine_path": "trt/parseq_m5_sage_b.engine",
    # },
    # {
    #     "id": "m5_sage_b_fuse_shapes",
    #     "label": "M5: Integer-Only PTQ + SageAttention (SAGEAttn-B) + Shapes",
    #     "variant": "m5",
    #     "ckpt": "pretrained/parseq_alpr_98.5.ckpt",
    #     "precision": "int8",
    #     "fusion_level": "shapes",
    #     "use_int_fa": False,
    #     "use_sage": True,
    #     "sage_mode": "sageattn_b",
    #     "onnx_path": "onnx/parseq_m5_sage_b_fuse_shapes.onnx",
    #     "engine_path": "trt/parseq_m5_sage_b_fuse_shapes.engine",
    # },
    # {
    #     "id": "m5_sage_b_fuse_mha",
    #     "label": "M5: Integer-Only PTQ + SageAttention (SAGEAttn-B) + Shapes + MHA",
    #     "variant": "m5",
    #     "ckpt": "pretrained/parseq_alpr_98.5.ckpt",
    #     "precision": "int8",
    #     "fusion_level": "mha",
    #     "use_int_fa": False,
    #     "use_sage": True,
    #     "sage_mode": "sageattn_b",
    #     "onnx_path": "onnx/parseq_m5_sage_b_fuse_mha.onnx",
    #     "engine_path": "trt/parseq_m5_sage_b_fuse_mha.engine",
    # },
    # {
    #     "id": "m5_sage_b_fuse_mlp",
    #     "label": "M5: Integer-Only PTQ + SageAttention (SAGEAttn-B) + Shapes + MHA + MLP",
    #     "variant": "m5",
    #     "ckpt": "pretrained/parseq_alpr_98.5.ckpt",
    #     "precision": "int8",
    #     "fusion_level": "mlp",
    #     "use_int_fa": False,
    #     "use_sage": True,
    #     "sage_mode": "sageattn_b",
    #     "onnx_path": "onnx/parseq_m5_sage_b_fuse_mlp.onnx",
    #     "engine_path": "trt/parseq_m5_sage_b_fuse_mlp.engine",
    # },
    # {
    #     "id": "m5_sage_b_fuse_all",
    #     "label": "M5: Integer-Only PTQ + SageAttention (SAGEAttn-B) + All Fused",
    #     "variant": "m5",
    #     "ckpt": "pretrained/parseq_alpr_98.5.ckpt",
    #     "precision": "int8",
    #     "fusion_level": "all",
    #     "use_int_fa": False,
    #     "use_sage": True,
    #     "sage_mode": "sageattn_b",
    #     "onnx_path": "onnx/parseq_m5_sage_b_fuse_all.onnx",
    #     "engine_path": "trt/parseq_m5_sage_b_fuse_all.engine",
    # },
    # Mode B: SAGEAttn-vB (Fully INT8 with per-channel V quantization) Progressive Fusion
    # {
    #     "id": "m5_sage_vb",
    #     "label": "M5: Integer-Only PTQ + SageAttention (SAGEAttn-vB Fully INT8)",
    #     "variant": "m5",
    #     "ckpt": "pretrained/parseq_alpr_98.5.ckpt",
    #     "precision": "int8",
    #     "fusion_level": "none",
    #     "use_int_fa": False,
    #     "use_sage": True,
    #     "sage_mode": "sageattn_vb",
    #     "v_quant_mode": "per_channel",
    #     "onnx_path": "onnx/parseq_m5_sage_vb.onnx",
    #     "engine_path": "trt/parseq_m5_sage_vb.engine",
    # },
    # {
    #     "id": "m5_sage_vb_fuse_shapes",
    #     "label": "M5: Integer-Only PTQ + SageAttention (SAGEAttn-vB) + Shapes",
    #     "variant": "m5",
    #     "ckpt": "pretrained/parseq_alpr_98.5.ckpt",
    #     "precision": "int8",
    #     "fusion_level": "shapes",
    #     "use_int_fa": False,
    #     "use_sage": True,
    #     "sage_mode": "sageattn_vb",
    #     "v_quant_mode": "per_channel",
    #     "onnx_path": "onnx/parseq_m5_sage_vb_fuse_shapes.onnx",
    #     "engine_path": "trt/parseq_m5_sage_vb_fuse_shapes.engine",
    # },
    # {
    #     "id": "m5_sage_vb_fuse_mha",
    #     "label": "M5: Integer-Only PTQ + SageAttention (SAGEAttn-vB) + Shapes + MHA",
    #     "variant": "m5",
    #     "ckpt": "pretrained/parseq_alpr_98.5.ckpt",
    #     "precision": "int8",
    #     "fusion_level": "mha",
    #     "use_int_fa": False,
    #     "use_sage": True,
    #     "sage_mode": "sageattn_vb",
    #     "v_quant_mode": "per_channel",
    #     "onnx_path": "onnx/parseq_m5_sage_vb_fuse_mha.onnx",
    #     "engine_path": "trt/parseq_m5_sage_vb_fuse_mha.engine",
    # },
    # {
    #     "id": "m5_sage_vb_fuse_mlp",
    #     "label": "M5: Integer-Only PTQ + SageAttention (SAGEAttn-vB) + Shapes + MHA + MLP",
    #     "variant": "m5",
    #     "ckpt": "pretrained/parseq_alpr_98.5.ckpt",
    #     "precision": "int8",
    #     "fusion_level": "mlp",
    #     "use_int_fa": False,
    #     "use_sage": True,
    #     "sage_mode": "sageattn_vb",
    #     "v_quant_mode": "per_channel",
    #     "onnx_path": "onnx/parseq_m5_sage_vb_fuse_mlp.onnx",
    #     "engine_path": "trt/parseq_m5_sage_vb_fuse_mlp.engine",
    # },
    # {
    #     "id": "m5_sage_vb_fuse_all",
    #     "label": "M5: Integer-Only PTQ + SageAttention (SAGEAttn-vB) + All Fused",
    #     "variant": "m5",
    #     "ckpt": "pretrained/parseq_alpr_98.5.ckpt",
    #     "precision": "int8",
    #     "fusion_level": "all",
    #     "use_int_fa": False,
    #     "use_sage": True,
    #     "sage_mode": "sageattn_vb",
    #     "v_quant_mode": "per_channel",
    #     "onnx_path": "onnx/parseq_m5_sage_vb_fuse_all.onnx",
    #     "engine_path": "trt/parseq_m5_sage_vb_fuse_all.engine",
    # },

    # =========================================================================
    # 7. M6: SageAttention (QAT - SAGEAttn-B & SAGEAttn-vB)
    # =========================================================================
    # Mode A: SAGEAttn-B (Float V) Progressive Fusion
    # {
    #     "id": "m6_sage_b",
    #     "label": "M6: Integer-Only QAT + SageAttention (SAGEAttn-B)",
    #     "variant": "m6",
    #     "ckpt": "pretrained/parseq_alpr_qat_m6.ckpt",
    #     "precision": "int8",
    #     "fusion_level": "none",
    #     "use_int_fa": False,
    #     "use_sage": True,
    #     "sage_mode": "sageattn_b",
    #     "onnx_path": "onnx/parseq_m6_sage_b.onnx",
    #     "engine_path": "trt/parseq_m6_sage_b.engine",
    # },
    # {
    #     "id": "m6_sage_b_fuse_shapes",
    #     "label": "M6: Integer-Only QAT + SageAttention (SAGEAttn-B) + Shapes",
    #     "variant": "m6",
    #     "ckpt": "pretrained/parseq_alpr_qat_m6.ckpt",
    #     "precision": "int8",
    #     "fusion_level": "shapes",
    #     "use_int_fa": False,
    #     "use_sage": True,
    #     "sage_mode": "sageattn_b",
    #     "onnx_path": "onnx/parseq_m6_sage_b_fuse_shapes.onnx",
    #     "engine_path": "trt/parseq_m6_sage_b_fuse_shapes.engine",
    # },
    # {
    #     "id": "m6_sage_b_fuse_mha",
    #     "label": "M6: Integer-Only QAT + SageAttention (SAGEAttn-B) + Shapes + MHA",
    #     "variant": "m6",
    #     "ckpt": "pretrained/parseq_alpr_qat_m6.ckpt",
    #     "precision": "int8",
    #     "fusion_level": "mha",
    #     "use_int_fa": False,
    #     "use_sage": True,
    #     "sage_mode": "sageattn_b",
    #     "onnx_path": "onnx/parseq_m6_sage_b_fuse_mha.onnx",
    #     "engine_path": "trt/parseq_m6_sage_b_fuse_mha.engine",
    # },
    # {
    #     "id": "m6_sage_b_fuse_mlp",
    #     "label": "M6: Integer-Only QAT + SageAttention (SAGEAttn-B) + Shapes + MHA + MLP",
    #     "variant": "m6",
    #     "ckpt": "pretrained/parseq_alpr_qat_m6.ckpt",
    #     "precision": "int8",
    #     "fusion_level": "mlp",
    #     "use_int_fa": False,
    #     "use_sage": True,
    #     "sage_mode": "sageattn_b",
    #     "onnx_path": "onnx/parseq_m6_sage_b_fuse_mlp.onnx",
    #     "engine_path": "trt/parseq_m6_sage_b_fuse_mlp.engine",
    # },
    # {
    #     "id": "m6_sage_b_fuse_all",
    #     "label": "M6: Integer-Only QAT + SageAttention (SAGEAttn-B) + All Fused",
    #     "variant": "m6",
    #     "ckpt": "pretrained/parseq_alpr_qat_m6.ckpt",
    #     "precision": "int8",
    #     "fusion_level": "all",
    #     "use_int_fa": False,
    #     "use_sage": True,
    #     "sage_mode": "sageattn_b",
    #     "onnx_path": "onnx/parseq_m6_sage_b_fuse_all.onnx",
    #     "engine_path": "trt/parseq_m6_sage_b_fuse_all.engine",
    # },
    # # Mode B: SAGEAttn-vB (Fully INT8) Progressive Fusion
    # {
    #     "id": "m6_sage_vb",
    #     "label": "M6: Integer-Only QAT + SageAttention (SAGEAttn-vB Fully INT8)",
    #     "variant": "m6",
    #     "ckpt": "pretrained/parseq_alpr_qat_m6.ckpt",
    #     "precision": "int8",
    #     "fusion_level": "none",
    #     "use_int_fa": False,
    #     "use_sage": True,
    #     "sage_mode": "sageattn_vb",
    #     "v_quant_mode": "per_channel",
    #     "onnx_path": "onnx/parseq_m6_sage_vb.onnx",
    #     "engine_path": "trt/parseq_m6_sage_vb.engine",
    # },
    # {
    #     "id": "m6_sage_vb_fuse_shapes",
    #     "label": "M6: Integer-Only QAT + SageAttention (SAGEAttn-vB) + Shapes",
    #     "variant": "m6",
    #     "ckpt": "pretrained/parseq_alpr_qat_m6.ckpt",
    #     "precision": "int8",
    #     "fusion_level": "shapes",
    #     "use_int_fa": False,
    #     "use_sage": True,
    #     "sage_mode": "sageattn_vb",
    #     "v_quant_mode": "per_channel",
    #     "onnx_path": "onnx/parseq_m6_sage_vb_fuse_shapes.onnx",
    #     "engine_path": "trt/parseq_m6_sage_vb_fuse_shapes.engine",
    # },
    # {
    #     "id": "m6_sage_vb_fuse_mha",
    #     "label": "M6: Integer-Only QAT + SageAttention (SAGEAttn-vB) + Shapes + MHA",
    #     "variant": "m6",
    #     "ckpt": "pretrained/parseq_alpr_qat_m6.ckpt",
    #     "precision": "int8",
    #     "fusion_level": "mha",
    #     "use_int_fa": False,
    #     "use_sage": True,
    #     "sage_mode": "sageattn_vb",
    #     "v_quant_mode": "per_channel",
    #     "onnx_path": "onnx/parseq_m6_sage_vb_fuse_mha.onnx",
    #     "engine_path": "trt/parseq_m6_sage_vb_fuse_mha.engine",
    # },
    # {
    #     "id": "m6_sage_vb_fuse_mlp",
    #     "label": "M6: Integer-Only QAT + SageAttention (SAGEAttn-vB) + Shapes + MHA + MLP",
    #     "variant": "m6",
    #     "ckpt": "pretrained/parseq_alpr_qat_m6.ckpt",
    #     "precision": "int8",
    #     "fusion_level": "mlp",
    #     "use_int_fa": False,
    #     "use_sage": True,
    #     "sage_mode": "sageattn_vb",
    #     "v_quant_mode": "per_channel",
    #     "onnx_path": "onnx/parseq_m6_sage_vb_fuse_mlp.onnx",
    #     "engine_path": "trt/parseq_m6_sage_vb_fuse_mlp.engine",
    # },
    # {
    #     "id": "m6_sage_vb_fuse_all",
    #     "label": "M6: Integer-Only QAT + SageAttention (SAGEAttn-vB) + All Fused",
    #     "variant": "m6",
    #     "ckpt": "pretrained/parseq_alpr_qat_m6.ckpt",
    #     "precision": "int8",
    #     "fusion_level": "all",
    #     "use_int_fa": False,
    #     "use_sage": True,
    #     "sage_mode": "sageattn_vb",
    #     "v_quant_mode": "per_channel",
    #     "onnx_path": "onnx/parseq_m6_sage_vb_fuse_all.onnx",
    #     "engine_path": "trt/parseq_m6_sage_vb_fuse_all.engine",
    # },

    # =========================================================================
    # 8. Custom Plugin Fused Variants (IPluginV2DynamicExt / CUDA DP4A)
    # =========================================================================
    # INT-FlashAttention Plugin
    # {
    #     "id": "m5_int_fa_plugin",
    #     "label": "M5: Integer-Only PTQ + INT-Flash Plugin",
    #     "variant": "m5",
    #     "ckpt": "pretrained/parseq_alpr_98.5.ckpt",
    #     "precision": "int8",
    #     "fusion_level": "none",
    #     "use_int_fa": True,
    #     "use_plugin": True,
    #     "onnx_path": "onnx/parseq_m5_int_fa_plugin.onnx",
    #     "engine_path": "trt/parseq_m5_int_fa_plugin.engine",
    # },
    # {
    #     "id": "m5_int_fa_plugin_fused",
    #     "label": "M5: Integer-Only PTQ + INT-Flash Plugin (All Fused)",
    #     "variant": "m5",
    #     "ckpt": "pretrained/parseq_alpr_98.5.ckpt",
    #     "precision": "int8",
    #     "fusion_level": "all",
    #     "use_int_fa": True,
    #     "use_plugin": True,
    #     "onnx_path": "onnx/parseq_m5_int_fa_plugin_fused.onnx",
    #     "engine_path": "trt/parseq_m5_int_fa_plugin_fused.engine",
    # },
    # {
    #     "id": "m6_int_fa_plugin",
    #     "label": "M6: Integer-Only QAT + INT-Flash Plugin",
    #     "variant": "m6",
    #     "ckpt": "pretrained/parseq_alpr_qat_m6.ckpt",
    #     "precision": "int8",
    #     "fusion_level": "none",
    #     "use_int_fa": True,
    #     "use_plugin": True,
    #     "onnx_path": "onnx/parseq_m6_int_fa_plugin.onnx",
    #     "engine_path": "trt/parseq_m6_int_fa_plugin.engine",
    # },
    # {
    #     "id": "m6_int_fa_plugin_fused",
    #     "label": "M6: Integer-Only QAT + INT-Flash Plugin (All Fused)",
    #     "variant": "m6",
    #     "ckpt": "pretrained/parseq_alpr_qat_m6.ckpt",
    #     "precision": "int8",
    #     "fusion_level": "all",
    #     "use_int_fa": True,
    #     "use_plugin": True,
    #     "onnx_path": "onnx/parseq_m6_int_fa_plugin_fused.onnx",
    #     "engine_path": "trt/parseq_m6_int_fa_plugin_fused.engine",
    # },
    # SageAttention Plugins
    # {
    #     "id": "m5_sage_b_plugin_fused",
    #     "label": "M5: Integer-Only PTQ + SageAttention Plugin (SAGEAttn-B, Fused)",
    #     "variant": "m5",
    #     "ckpt": "pretrained/parseq_alpr_98.5.ckpt",
    #     "precision": "int8",
    #     "fusion_level": "all",
    #     "use_int_fa": False,
    #     "use_sage": True,
    #     "sage_mode": "sageattn_b",
    #     "use_plugin": True,
    #     "onnx_path": "onnx/parseq_m5_sage_b_plugin_fused.onnx",
    #     "engine_path": "trt/parseq_m5_sage_b_plugin_fused.engine",
    # },
    # {
    #     "id": "m5_sage_vb_plugin_fused",
    #     "label": "M5: Integer-Only PTQ + SageAttention Plugin (SAGEAttn-vB, Fused)",
    #     "variant": "m5",
    #     "ckpt": "pretrained/parseq_alpr_98.5.ckpt",
    #     "precision": "int8",
    #     "fusion_level": "all",
    #     "use_int_fa": False,
    #     "use_sage": True,
    #     "sage_mode": "sageattn_vb",
    #     "v_quant_mode": "per_channel",
    #     "use_plugin": True,
    #     "onnx_path": "onnx/parseq_m5_sage_vb_plugin_fused.onnx",
    #     "engine_path": "trt/parseq_m5_sage_vb_plugin_fused.engine",
    # },
    # {
    #     "id": "m6_sage_b_plugin_fused",
    #     "label": "M6: Integer-Only QAT + SageAttention Plugin (SAGEAttn-B, Fused)",
    #     "variant": "m6",
    #     "ckpt": "pretrained/parseq_alpr_qat_m6.ckpt",
    #     "precision": "int8",
    #     "fusion_level": "all",
    #     "use_int_fa": False,
    #     "use_sage": True,
    #     "sage_mode": "sageattn_b",
    #     "use_plugin": True,
    #     "onnx_path": "onnx/parseq_m6_sage_b_plugin_fused.onnx",
    #     "engine_path": "trt/parseq_m6_sage_b_plugin_fused.engine",
    # },
    # {
    #     "id": "m6_sage_vb_plugin_fused",
    #     "label": "M6: Integer-Only QAT + SageAttention Plugin (SAGEAttn-vB, Fused)",
    #     "variant": "m6",
    #     "ckpt": "pretrained/parseq_alpr_qat_m6.ckpt",
    #     "precision": "int8",
    #     "fusion_level": "all",
    #     "use_int_fa": False,
    #     "use_sage": True,
    #     "sage_mode": "sageattn_vb",
    #     "v_quant_mode": "per_channel",
    #     "use_plugin": True,
    #     "onnx_path": "onnx/parseq_m6_sage_vb_plugin_fused.onnx",
    #     "engine_path": "trt/parseq_m6_sage_vb_plugin_fused.engine",
    # },
    # All Custom Plugins (FA + LN + GELU + Softmax)
    # {
    #     "id": "m5_plugin_all",
    #     "label": "M5: Integer-Only PTQ + All Custom Plugins (FA+LN+GELU)",
    #     "variant": "m5",
    #     "ckpt": "pretrained/parseq_alpr_98.5.ckpt",
    #     "precision": "int8",
    #     "fusion_level": "none",
    #     "use_int_fa": True,
    #     "use_plugin": True,
    #     "onnx_path": "onnx/parseq_m5_plugin_all.onnx",
    #     "engine_path": "trt/parseq_m5_plugin_all.engine",
    # },
    # {
    #     "id": "m5_plugin_all_fused",
    #     "label": "M5: Integer-Only PTQ + All Custom Plugins (All Fused)",
    #     "variant": "m5",
    #     "ckpt": "pretrained/parseq_alpr_98.5.ckpt",
    #     "precision": "int8",
    #     "fusion_level": "all",
    #     "use_int_fa": True,
    #     "use_plugin": True,
    #     "onnx_path": "onnx/parseq_m5_plugin_all_fused.onnx",
    #     "engine_path": "trt/parseq_m5_plugin_all_fused.engine",
    # },
    # {
    #     "id": "m6_plugin_all",
    #     "label": "M6: Integer-Only QAT + All Custom Plugins (FA+LN+GELU)",
    #     "variant": "m6",
    #     "ckpt": "pretrained/parseq_alpr_qat_m6.ckpt",
    #     "precision": "int8",
    #     "fusion_level": "none",
    #     "use_int_fa": True,
    #     "use_plugin": True,
    #     "onnx_path": "onnx/parseq_m6_plugin_all.onnx",
    #     "engine_path": "trt/parseq_m6_plugin_all.engine",
    # },
    # {
    #     "id": "m6_plugin_all_fused",
    #     "label": "M6: Integer-Only QAT + All Custom Plugins (All Fused)",
    #     "variant": "m6",
    #     "ckpt": "pretrained/parseq_alpr_qat_m6.ckpt",
    #     "precision": "int8",
    #     "fusion_level": "all",
    #     "use_int_fa": True,
    #     "use_plugin": True,
    #     "onnx_path": "onnx/parseq_m6_plugin_all_fused.onnx",
    #     "engine_path": "trt/parseq_m6_plugin_all_fused.engine",
    # },

    # =========================================================================
    # 9. Alternative Integer Nonlinear Operators (IViT Shiftmax & I-BERT)
    # =========================================================================
    {
        "id": "m5_ivit_approx",
        "label": "M5: Integer-Only PTQ + IViT Approximations (Shiftmax/IViTGELU)",
        "variant": "m5",
        "ckpt": "pretrained/parseq_alpr_98.5.ckpt",
        "precision": "int8",
        "fusion_level": "none",
        "use_int_fa": False,
        "gelu_candidate": "gelu_ivit",
        "softmax_candidate": "softmax_ivit",
        "layernorm_candidate": "layernorm_ibert",
        "onnx_path": "onnx/parseq_m5_ivit.onnx",
        "engine_path": "trt/parseq_m5_ivit.engine",
    },
    {
        "id": "m5_ivit_approx_fused",
        "label": "M5: Integer-Only PTQ + IViT Approximations (All Fused)",
        "variant": "m5",
        "ckpt": "pretrained/parseq_alpr_98.5.ckpt",
        "precision": "int8",
        "fusion_level": "all",
        "use_int_fa": False,
        "gelu_candidate": "gelu_ivit",
        "softmax_candidate": "softmax_ivit",
        "layernorm_candidate": "layernorm_ibert",
        "onnx_path": "onnx/parseq_m5_ivit_fused.onnx",
        "engine_path": "trt/parseq_m5_ivit_fused.engine",
    },
    {
        "id": "m5_ibert_approx",
        "label": "M5: Integer-Only PTQ + I-BERT Approximations",
        "variant": "m5",
        "ckpt": "pretrained/parseq_alpr_98.5.ckpt",
        "precision": "int8",
        "fusion_level": "none",
        "use_int_fa": False,
        "gelu_candidate": "gelu_ibert",
        "softmax_candidate": "softmax_ibert",
        "layernorm_candidate": "layernorm_ibert",
        "onnx_path": "onnx/parseq_m5_ibert.onnx",
        "engine_path": "trt/parseq_m5_ibert.engine",
    },
    {
        "id": "m5_ibert_approx_fused",
        "label": "M5: Integer-Only PTQ + I-BERT Approximations (All Fused)",
        "variant": "m5",
        "ckpt": "pretrained/parseq_alpr_98.5.ckpt",
        "precision": "int8",
        "fusion_level": "all",
        "use_int_fa": False,
        "gelu_candidate": "gelu_ibert",
        "softmax_candidate": "softmax_ibert",
        "layernorm_candidate": "layernorm_ibert",
        "onnx_path": "onnx/parseq_m5_ibert_fused.onnx",
        "engine_path": "trt/parseq_m5_ibert_fused.engine",
    },
    {
        "id": "m6_ivit_approx",
        "label": "M6: Integer-Only QAT + IViT Approximations (Shiftmax/IViTGELU)",
        "variant": "m6",
        "ckpt": "pretrained/parseq_alpr_qat_m6.ckpt",
        "precision": "int8",
        "fusion_level": "none",
        "use_int_fa": False,
        "gelu_candidate": "gelu_ivit",
        "softmax_candidate": "softmax_ivit",
        "layernorm_candidate": "layernorm_ibert",
        "onnx_path": "onnx/parseq_m6_ivit.onnx",
        "engine_path": "trt/parseq_m6_ivit.engine",
    },
    {
        "id": "m6_ivit_approx_fused",
        "label": "M6: Integer-Only QAT + IViT Approximations (All Fused)",
        "variant": "m6",
        "ckpt": "pretrained/parseq_alpr_qat_m6.ckpt",
        "precision": "int8",
        "fusion_level": "all",
        "use_int_fa": False,
        "gelu_candidate": "gelu_ivit",
        "softmax_candidate": "softmax_ivit",
        "layernorm_candidate": "layernorm_ibert",
        "onnx_path": "onnx/parseq_m6_ivit_fused.onnx",
        "engine_path": "trt/parseq_m6_ivit_fused.engine",
    },
    {
        "id": "m6_ibert_approx",
        "label": "M6: Integer-Only QAT + I-BERT Approximations",
        "variant": "m6",
        "ckpt": "pretrained/parseq_alpr_qat_m6.ckpt",
        "precision": "int8",
        "fusion_level": "none",
        "use_int_fa": False,
        "gelu_candidate": "gelu_ibert",
        "softmax_candidate": "softmax_ibert",
        "layernorm_candidate": "layernorm_ibert",
        "onnx_path": "onnx/parseq_m6_ibert.onnx",
        "engine_path": "trt/parseq_m6_ibert.engine",
    },
    {
        "id": "m6_ibert_approx_fused",
        "label": "M6: Integer-Only QAT + I-BERT Approximations (All Fused)",
        "variant": "m6",
        "ckpt": "pretrained/parseq_alpr_qat_m6.ckpt",
        "precision": "int8",
        "fusion_level": "all",
        "use_int_fa": False,
        "gelu_candidate": "gelu_ibert",
        "softmax_candidate": "softmax_ibert",
        "layernorm_candidate": "layernorm_ibert",
        "onnx_path": "onnx/parseq_m6_ibert_fused.onnx",
        "engine_path": "trt/parseq_m6_ibert_fused.engine",
    },
]


def run_full_pipeline(
    max_eval_samples: int = 50,
    force: bool = False,
    checkpoint: str = "pretrained/parseq_alpr_98.5.ckpt",
    batch_size: int = 64,
    num_workers: int = 0,
    workspace_gb: float = 4.0,
    config_filter: Optional[str] = None,
) -> Dict[str, Any]:
    os.makedirs("onnx", exist_ok=True)
    os.makedirs("trt", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    summary_records = []

    # Prepare base system for ALPR loader
    base_sys = load_from_checkpoint(checkpoint).eval()
    hp = base_sys.hparams
    datamodule = SceneTextDataModule(
        root_dir="data",
        train_dir="_unused_",
        img_size=hp.img_size,
        max_label_length=hp.max_label_length,
        charset_train=hp.charset_train,
        charset_test=hp.charset_test,
        batch_size=batch_size,
        num_workers=num_workers,
        augment=False,
    )
    test_loader = datamodule.test_dataloaders(["VeSV_pad"])["VeSV_pad"]

    # batch_sizes = [1 << i for i in range((max(1, batch_size)).bit_length()) if (1 << i) <= batch_size]
    batch_sizes = [1, 128, 256]
    if not batch_sizes:
        batch_sizes = [1]

    active_configs = CONFIGURATIONS
    if config_filter:
        flt = config_filter.lower().strip()
        active_configs = [c for c in CONFIGURATIONS if flt in c["id"].lower() or flt in c["label"].lower()]
        print(f"Filter '{config_filter}' applied: {len(active_configs)}/{len(CONFIGURATIONS)} configurations selected.")
        if not active_configs:
            print(f"Warning: No configurations matched filter '{config_filter}'.")
            return []

    print("=" * 80)
    print(f"STARTING FULL MATRIX EXECUTION: {len(active_configs)} CONFIGURATIONS (Batches: {batch_sizes})")
    print("=" * 80)

    for idx, cfg in enumerate(active_configs, 1):
        print(f"\n[{idx}/{len(active_configs)}] Processing: {cfg['label']}")
        print(f"      ONNX:   {cfg['onnx_path']}")
        print(f"      Engine: {cfg['engine_path']}")

        # Determine descriptive attention / operator type
        if cfg.get("use_plugin") and "plugin_all" in cfg["id"]:
            attn_type = "Custom Plugins (All: FA+LN+GELU)"
        elif cfg.get("use_plugin") and cfg.get("use_sage"):
            attn_type = f"SagePlugin ({'vB' if cfg.get('sage_mode') == 'sageattn_vb' else 'B'})"
        elif cfg.get("use_plugin") and cfg.get("use_int_fa"):
            attn_type = "INT-Flash Plugin"
        elif cfg.get("use_sage"):
            attn_type = f"SageAttn ({'vB' if cfg.get('sage_mode') == 'sageattn_vb' else 'B'})"
        elif cfg.get("use_int_fa"):
            attn_type = "INT-Flash"
        elif "ivit" in cfg["id"]:
            attn_type = "IViT Shiftmax"
        elif "ibert" in cfg["id"]:
            attn_type = "I-BERT Softmax"
        else:
            attn_type = "Padrão"

        record = {
            "id": cfg["id"],
            "label": cfg["label"],
            "variant": cfg["variant"],
            "precision": cfg["precision"],
            "fusion": cfg["fusion_level"],
            "attn_type": attn_type,
            "int_fa": cfg.get("use_int_fa", False),
            "sage_attn": cfg.get("use_sage", False),
            "sage_mode": cfg.get("sage_mode", "none"),
            "use_plugin": cfg.get("use_plugin", False),
            "onnx_size_mb": None,
            "engine_size_mb": None,
            "latency_b1_mean": None,
            "latency_b1_p95": None,
            "fps_b1": None,
            "latency_b32_mean": None,
            "fps_b32": None,
            "exact_acc": None,
            "ned": None,
            "cer": None,
            "status": "PENDING",
        }

        # Check if a specific variant checkpoint exists; otherwise fallback to normal/default
        cfg_id = cfg["id"]
        specific_candidates = [
            f"pretrained/parseq_alpr_qat_{cfg_id}.ckpt",
            f"pretrained/parseq_alpr_{cfg_id}.ckpt",
            f"pretrained/{cfg_id}.ckpt",
            f"pretrained/{cfg_id}_best.ckpt",
            f"checkpoints/parseq_alpr_qat_{cfg_id}.ckpt",
            f"checkpoints/{cfg_id}.ckpt",
        ]
        chosen_ckpt = cfg["ckpt"]
        is_specific = False
        for cand in specific_candidates:
            if os.path.exists(cand):
                chosen_ckpt = cand
                is_specific = True
                break

        if is_specific:
            print(f"      [Checkpoint] Specific checkpoint detected: {chosen_ckpt}")
        else:
            print(f"      [Checkpoint] Using default checkpoint: {chosen_ckpt}")

        record["ckpt"] = chosen_ckpt
        record["is_specific_ckpt"] = is_specific

        # Step 1: Export ONNX
        try:
            ln_py = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "strhub", "quant", "integer_layernorm.py")
            ln_mtime = os.path.getmtime(ln_py) if os.path.exists(ln_py) else 0
            onnx_exists = os.path.exists(cfg["onnx_path"])
            onnx_mtime = os.path.getmtime(cfg["onnx_path"]) if onnx_exists else 0

            onnx_needs_rebuild = (
                not onnx_exists
                or force
                or (onnx_exists and ln_mtime > onnx_mtime)
                or (is_specific and os.path.exists(chosen_ckpt) and onnx_exists and os.path.getmtime(chosen_ckpt) > onnx_mtime)
            )
            if onnx_needs_rebuild:
                print("   -> Exporting ONNX...")
                fuse_shapes = cfg["fusion_level"] in ["shapes", "mha", "mlp", "all"]
                fuse_mha = cfg["fusion_level"] in ["mha", "mlp", "all"]
                fuse_mlp = cfg["fusion_level"] in ["mlp", "all"]
                fuse_layernorm = cfg["fusion_level"] in ["all"]

                export_onnx(
                    checkpoint_path=chosen_ckpt,
                    variant=cfg["variant"],
                    output_path=cfg["onnx_path"],
                    opset_version=17,
                    fuse_shapes=fuse_shapes,
                    fuse_mha=fuse_mha,
                    fuse_mlp=fuse_mlp,
                    fuse_layernorm=fuse_layernorm,
                    use_int_flashattention=cfg.get("use_int_fa", False),
                    use_sage_attention=cfg.get("use_sage", False),
                    sage_mode=cfg.get("sage_mode", "sageattn_b"),
                    v_quant_mode=cfg.get("v_quant_mode", "per_tensor"),
                    gelu_candidate=cfg.get("gelu_candidate", "gelu_iptq"),
                    softmax_candidate=cfg.get("softmax_candidate", "softmax_iptq"),
                    layernorm_candidate=cfg.get("layernorm_candidate", "layernorm_ibert"),
                    use_plugin=cfg.get("use_plugin", False),
                )
            record["onnx_size_mb"] = round(os.path.getsize(cfg["onnx_path"]) / (1024 * 1024), 2)
            print(f"   ✓ ONNX ready ({record['onnx_size_mb']} MB)")
        except Exception as e:
            print(f"   ✗ Export failed: {e}")
            record["status"] = f"EXPORT_FAILED: {e}"
            summary_records.append(record)
            continue

        # Step 2: Build TensorRT Engine
        try:
            ws_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            trt_plugins_py = os.path.join(ws_root, "strhub", "quant", "plugins", "trt_plugins.py")
            plugins_so = os.path.join(ws_root, "strhub", "quant", "plugins", "parseq_plugins.so")
            plugin_mtime = max(
                os.path.getmtime(trt_plugins_py) if os.path.exists(trt_plugins_py) else 0,
                os.path.getmtime(plugins_so) if os.path.exists(plugins_so) else 0,
            )
            engine_exists = os.path.exists(cfg["engine_path"])
            engine_mtime = os.path.getmtime(cfg["engine_path"]) if engine_exists else 0

            engine_needs_rebuild = (
                not engine_exists
                or force
                or onnx_needs_rebuild
                or (cfg.get("use_plugin") and engine_exists and plugin_mtime > engine_mtime)
                or (is_specific and os.path.exists(chosen_ckpt) and engine_exists and os.path.getmtime(chosen_ckpt) > engine_mtime)
            )
            if engine_needs_rebuild:
                if cfg.get("use_plugin", False):
                    from strhub.quant.plugins.trt_plugins import find_plugin_lib_path
                    plugin_lib = find_plugin_lib_path()
                    if not os.path.isfile(plugin_lib):
                        print(f"   [!] Plugin library missing at {plugin_lib}. Attempting automatic compilation...")
                        try:
                            from tools.build_plugins import compile_cuda_plugins
                            compile_cuda_plugins(verbose=True)
                        except Exception as ce:
                            print(f"   [!] Auto-compilation notice: {ce}")
                print(f"   -> Building TensorRT Engine ({cfg['precision'].upper()})...")
                build_tensorrt_engine(
                    onnx_path=cfg["onnx_path"],
                    engine_path=cfg["engine_path"],
                    precision=cfg["precision"],
                    max_batch_size=batch_size,
                    workspace_gb=workspace_gb,
                )
            record["engine_size_mb"] = round(os.path.getsize(cfg["engine_path"]) / (1024 * 1024), 2)
            print(f"   ✓ Engine ready ({record['engine_size_mb']} MB)")
        except Exception as e:
            print(f"   ✗ Build failed: {e}")
            record["status"] = f"BUILD_FAILED: {e}"
            summary_records.append(record)
            continue

        # Step 3: Benchmark TensorRT Engine across powers of 2 up to batch_size (1 << i)
        try:
            max_fps = -1.0
            max_fps_batch = 1
            for b in batch_sizes:
                print(f"   -> Running TensorRT Benchmark (Batch={b})...")
                res_b = benchmark_tensorrt(
                    cfg["engine_path"],
                    batch_size=b,
                    num_warmup=20 if b == 1 else 10,
                    num_iterations=100 if b == 1 else 50,
                )
                record["layers"] = f"Compiled Layers: {res_b.get('total_layers', 'N/A')} {res_b.get('layer_types', {})}"
                record[f"latency_b{b}_mean"] = round(res_b["mean_ms"], 2)
                record[f"fps_b{b}"] = round(res_b["fps"], 1)
                if b == 1:
                    record["latency_b1_p95"] = round(res_b["p95_ms"], 2)

                if res_b["fps"] > max_fps:
                    max_fps = res_b["fps"]
                    max_fps_batch = b

            record["max_fps"] = round(max_fps, 1)
            record["max_fps_batch"] = max_fps_batch

            print(
                f"   ✓ B1 Latency: {record.get('latency_b1_mean', 0.0)}ms ({record.get('fps_b1', 0.0)} FPS) | "
                f"B{max_fps_batch} (Max): {record.get(f'latency_b{max_fps_batch}_mean', 0.0)}ms ({record.get(f'fps_b{max_fps_batch}', 0.0)} FPS)"
            )
            print(record["layers"])
        except Exception as e:
            print(f"   ✗ Benchmark failed: {e}")
            record["status"] = f"BENCH_FAILED: {e}"

        # Step 4: Validate ALPR OCR Accuracy
        trt_model = None
        try:
            print(f"   -> Evaluating ALPR Accuracy on VeSV_pad ({max_eval_samples} samples)...")
            trt_model = TensorRTModelWrapper(cfg["engine_path"], device="cuda")
            acc_metrics = evaluate_dataset(
                model=trt_model,
                data_loader=test_loader,
                tokenizer=base_sys.tokenizer,
                charset_adapter=base_sys.charset_adapter,
                device=torch.device("cuda"),
                max_samples=max_eval_samples,
            )
            record["exact_acc"] = round(acc_metrics["exact_plate_accuracy"] * 100, 2)
            record["ned"] = round(acc_metrics["normalized_edit_distance"] * 100, 2)
            record["cer"] = round(acc_metrics["character_error_rate"] * 100, 2)
            record["status"] = "SUCCESS"
            print(f"   ✓ Exact Plate Accuracy: {record['exact_acc']}% | NED: {record['ned']}% | CER: {record['cer']}%")
        except Exception as e:
            print(f"   ✗ Accuracy evaluation failed: {e}")
            if record["status"] == "PENDING":
                record["status"] = f"EVAL_FAILED: {e}"
        finally:
            if trt_model is not None:
                if hasattr(trt_model, "close"):
                    trt_model.close()
                del trt_model
                torch.cuda.empty_cache()
                import gc
                gc.collect()

        summary_records.append(record)

    # Step 5: Save JSON Report
    report_json_path = "results/full_matrix_benchmark_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_records, f, indent=2)

    # Step 6: Generate Markdown Table
    report_md_path = "results/full_matrix_benchmark_report.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# Relatório Comparativo Completo: Matriz de Variantes PARSeq\n\n")
        headers = ["ID", "Variante", "Precisão", "Fusão", "Atenção / Operador", "Checkpoint", "Engine (MB)"]
        for b in batch_sizes:
            headers.extend([f"Latência B{b} (ms)", f"FPS B{b}"])
        headers.extend(["Maior FPS (Batch)", "Acurácia Placa (%)", "NED (%)", "Status"])

        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")

        for r in summary_records:
            attn_type = r.get("attn_type", "Padrão")
            max_fps_str = (
                f"{r['max_fps']} (B{r['max_fps_batch']})"
                if r.get("max_fps") is not None
                else "-"
            )
            acc_str = f"{r['exact_acc']}%" if r.get("exact_acc") is not None else "-"
            ned_str = f"{r['ned']}%" if r.get("ned") is not None else "-"

            ckpt_name = os.path.basename(r.get("ckpt", ""))
            ckpt_str = f"**{ckpt_name}** (Específico)" if r.get("is_specific_ckpt") else ckpt_name

            row = [
                f"`{r['id']}`",
                str(r["label"]),
                str(r["precision"].upper()),
                str(r["fusion"]),
                str(attn_type),
                str(ckpt_str),
                str(r.get("engine_size_mb", "-")),
            ]
            for b in batch_sizes:
                row.append(str(r.get(f"latency_b{b}_mean", "-")))
                row.append(str(r.get(f"fps_b{b}", "-")))
            row.extend([max_fps_str, acc_str, ned_str, str(r.get("status", "-"))])

            f.write("| " + " | ".join(row) + " |\n")

    print("\n" + "=" * 80)
    print("ALL PIPELINE STAGES COMPLETED!")
    print(f"Results saved to:\n  - {report_json_path}\n  - {report_md_path}")
    print("=" * 80)

    return summary_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pos_checkpoint", nargs="?", default=None, help="Optional positional checkpoint ckpt model")
    parser.add_argument("--checkpoint", type=str, default="pretrained/parseq_alpr_98.5.ckpt", help="Checkpoint ckpt model")
    parser.add_argument("--batch_size", type=int, default=64, help="Maximum batch size for engine compilation and benchmarking (powers of 2: 1, 2, 4, ...)")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of DataLoader workers")
    parser.add_argument("--samples", type=int, default=50, help="Number of ALPR samples to evaluate")
    parser.add_argument("--workspace_gb", type=float, default=4.0, help="Workspace memory limit in GB for TensorRT builder")
    parser.add_argument("--filter", type=str, default=None, help="Filter configurations by substring matching id or label")
    parser.add_argument("--force", action="store_true", help="Force re-export and rebuild of all engines")
    args = parser.parse_args()

    ckpt = args.pos_checkpoint if args.pos_checkpoint is not None else args.checkpoint
    run_full_pipeline(
        max_eval_samples=args.samples,
        force=args.force,
        checkpoint=ckpt,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        workspace_gb=args.workspace_gb,
        config_filter=args.filter,
    )
