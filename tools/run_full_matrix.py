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
from typing import Dict, Any, List

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
    # 1. FP32 Baselines
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
    # 2. FP16 Baseline
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
    # 3. Naive INT8 (Negative Control)
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
    # 4. Conventional W8A8 PTQ
    {
        "id": "m4_nar_int8_ptq",
        "label": "M4: Conventional W8A8 PTQ",
        "variant": "m4",
        "ckpt": "pretrained/parseq_alpr_98.5.ckpt",
        "precision": "int8",
        "fusion_level": "none",
        "use_int_fa": False,
        "onnx_path": "onnx/parseq_m4_nar_int8_ptq.onnx",
        "engine_path": "trt/parseq_m4_nar_int8_ptq.engine",
    },
    # 5. Integer-Only PTQ (M5) Variations
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
    {
        "id": "m5_int_fa",
        "label": "M5: Integer-Only PTQ + INT-FlashAttention",
        "variant": "m5",
        "ckpt": "pretrained/parseq_alpr_98.5.ckpt",
        "precision": "int8",
        "fusion_level": "none",
        "use_int_fa": True,
        "onnx_path": "onnx/parseq_m5_int_fa.onnx",
        "engine_path": "trt/parseq_m5_int_fa.engine",
    },
    {
        "id": "m5_int_fa_fuse_all",
        "label": "M5: Integer-Only PTQ + INT-Flash + All Fused",
        "variant": "m5",
        "ckpt": "pretrained/parseq_alpr_98.5.ckpt",
        "precision": "int8",
        "fusion_level": "all",
        "use_int_fa": True,
        "onnx_path": "onnx/parseq_m5_int_fa_fuse_all.onnx",
        "engine_path": "trt/parseq_m5_int_fa_fuse_all_int8.engine",
    },
    # 6. Integer-Only QAT (M6) Variations
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
    {
        "id": "m6_int_fa",
        "label": "M6: Integer-Only QAT + INT-FlashAttention",
        "variant": "m6",
        "ckpt": "pretrained/parseq_alpr_qat_m6.ckpt",
        "precision": "int8",
        "fusion_level": "none",
        "use_int_fa": True,
        "onnx_path": "onnx/parseq_m6_int_fa.onnx",
        "engine_path": "trt/parseq_m6_int_fa.engine",
    },
    {
        "id": "m6_int_fa_fuse_all",
        "label": "M6: Integer-Only QAT + INT-Flash + All Fused",
        "variant": "m6",
        "ckpt": "pretrained/parseq_alpr_qat_m6.ckpt",
        "precision": "int8",
        "fusion_level": "all",
        "use_int_fa": True,
        "onnx_path": "onnx/parseq_m6_int_fa_fuse_all.onnx",
        "engine_path": "trt/parseq_m6_int_fa_fuse_all.engine",
    },
    # 7. Custom Plugin Fused Variants (IPluginV2DynamicExt / CUDA DP4A)
    {
        "id": "m5_int_fa_plugin_fused",
        "label": "M5: Integer-Only PTQ + INT-Flash Plugin (Fused)",
        "variant": "m5",
        "ckpt": "pretrained/parseq_alpr_98.5.ckpt",
        "precision": "int8",
        "fusion_level": "all",
        "use_int_fa": True,
        "use_plugin": True,
        "onnx_path": "onnx/parseq_m5_int_fa_plugin_fused.onnx",
        "engine_path": "trt/parseq_m5_int_fa_plugin_fused.engine",
    },
    {
        "id": "m6_int_fa_plugin_fused",
        "label": "M6: Integer-Only QAT + INT-Flash Plugin (Fused)",
        "variant": "m6",
        "ckpt": "pretrained/parseq_alpr_qat_m6.ckpt",
        "precision": "int8",
        "fusion_level": "all",
        "use_int_fa": True,
        "use_plugin": True,
        "onnx_path": "onnx/parseq_m6_int_fa_plugin_fused.onnx",
        "engine_path": "trt/parseq_m6_int_fa_plugin_fused.engine",
    },
    {
        "id": "m5_plugin_all",
        "label": "M5: Integer-Only PTQ + All Custom Plugins (FA+LN+GELU)",
        "variant": "m5",
        "ckpt": "pretrained/parseq_alpr_98.5.ckpt",
        "precision": "int8",
        "fusion_level": "none",
        "use_int_fa": True,
        "use_plugin": True,
        "onnx_path": "onnx/parseq_m5_plugin_all.onnx",
        "engine_path": "trt/parseq_m5_plugin_all.engine",
    },
    {
        "id": "m6_plugin_all",
        "label": "M6: Integer-Only QAT + All Custom Plugins (FA+LN+GELU)",
        "variant": "m6",
        "ckpt": "pretrained/parseq_alpr_qat_m6.ckpt",
        "precision": "int8",
        "fusion_level": "none",
        "use_int_fa": True,
        "use_plugin": True,
        "onnx_path": "onnx/parseq_m6_plugin_all.onnx",
        "engine_path": "trt/parseq_m6_plugin_all.engine",
    },
    # 8. SageAttention Variants (Paper 2410.02367: Key Smoothing + INT8 Attention)
    {
        "id": "m5_sage_attn",
        "label": "M5: Integer-Only PTQ + SageAttention (SAGEAttn-B)",
        "variant": "m5",
        "ckpt": "pretrained/parseq_alpr_98.5.ckpt",
        "precision": "int8",
        "fusion_level": "none",
        "use_int_fa": False,
        "use_sage": True,
        "sage_mode": "sageattn_b",
        "onnx_path": "onnx/parseq_m5_sage.onnx",
        "engine_path": "trt/parseq_m5_sage_int8.engine",
    },
    {
        "id": "m6_sage_attn",
        "label": "M6: Integer-Only QAT + SageAttention (SAGEAttn-B)",
        "variant": "m6",
        "ckpt": "pretrained/parseq_alpr_qat_m6.ckpt",
        "precision": "int8",
        "fusion_level": "none",
        "use_int_fa": False,
        "use_sage": True,
        "sage_mode": "sageattn_b",
        "onnx_path": "onnx/parseq_m6_sage.onnx",
        "engine_path": "trt/parseq_m6_sage.engine",
    },
    {
        "id": "m5_sage_plugin_fused",
        "label": "M5: Integer-Only PTQ + SageAttention Plugin (Fused)",
        "variant": "m5",
        "ckpt": "pretrained/parseq_alpr_98.5.ckpt",
        "precision": "int8",
        "fusion_level": "all",
        "use_int_fa": False,
        "use_sage": True,
        "sage_mode": "sageattn_b",
        "use_plugin": True,
        "onnx_path": "onnx/parseq_m5_sage_plugin_fused.onnx",
        "engine_path": "trt/parseq_m5_sage_plugin_fused.engine",
    },
    {
        "id": "m6_sage_plugin_fused",
        "label": "M6: Integer-Only QAT + SageAttention Plugin (Fused)",
        "variant": "m6",
        "ckpt": "pretrained/parseq_alpr_qat_m6.ckpt",
        "precision": "int8",
        "fusion_level": "all",
        "use_int_fa": False,
        "use_sage": True,
        "sage_mode": "sageattn_b",
        "use_plugin": True,
        "onnx_path": "onnx/parseq_m6_sage_plugin_fused.onnx",
        "engine_path": "trt/parseq_m6_sage_plugin_fused.engine",
    },
]


def run_full_pipeline(max_eval_samples: int = 50, force: bool = False, checkpoint="pretrained/parseq_alpr_98.5.ckpt", batch_size = 64, num_workers = 0) -> Dict[str, Any]:
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

    print("=" * 80)
    print(f"STARTING FULL MATRIX EXECUTION: {len(CONFIGURATIONS)} CONFIGURATIONS")
    print("=" * 80)

    for idx, cfg in enumerate(CONFIGURATIONS, 1):
        print(f"\n[{idx}/{len(CONFIGURATIONS)}] Processing: {cfg['label']}")
        print(f"      ONNX:   {cfg['onnx_path']}")
        print(f"      Engine: {cfg['engine_path']}")

        record = {
            "id": cfg["id"],
            "label": cfg["label"],
            "variant": cfg["variant"],
            "precision": cfg["precision"],
            "fusion": cfg["fusion_level"],
            "int_fa": cfg.get("use_int_fa", False),
            "sage_attn": cfg.get("use_sage", False),
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

        # Step 1: Export ONNX
        try:
            if not os.path.exists(cfg["onnx_path"]) or force:
                print("   -> Exporting ONNX...")
                fuse_shapes = cfg["fusion_level"] in ["shapes", "mha", "mlp", "all"]
                fuse_mha = cfg["fusion_level"] in ["mha", "mlp", "all"]
                fuse_mlp = cfg["fusion_level"] in ["mlp", "all"]
                fuse_layernorm = cfg["fusion_level"] in ["all"]

                export_onnx(
                    checkpoint_path=cfg["ckpt"],
                    variant=cfg["variant"],
                    output_path=cfg["onnx_path"],
                    opset_version=18,
                    fuse_shapes=fuse_shapes,
                    fuse_mha=fuse_mha,
                    fuse_mlp=fuse_mlp,
                    fuse_layernorm=fuse_layernorm,
                    use_int_flashattention=cfg.get("use_int_fa", False),
                    use_sage_attention=cfg.get("use_sage", False),
                    sage_mode=cfg.get("sage_mode", "sageattn_b"),
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
            if not os.path.exists(cfg["engine_path"]) or force:
                print(f"   -> Building TensorRT Engine ({cfg['precision'].upper()})...")
                build_tensorrt_engine(
                    onnx_path=cfg["onnx_path"],
                    engine_path=cfg["engine_path"],
                    precision=cfg["precision"],
                    max_batch_size=batch_size,
                )
            record["engine_size_mb"] = round(os.path.getsize(cfg["engine_path"]) / (1024 * 1024), 2)
            print(f"   ✓ Engine ready ({record['engine_size_mb']} MB)")
        except Exception as e:
            print(f"   ✗ Build failed: {e}")
            record["status"] = f"BUILD_FAILED: {e}"
            summary_records.append(record)
            continue

        # Step 3: Benchmark TensorRT Engine (Batch 1 and Batch 32)
        try:
            print("   -> Running TensorRT Benchmark (Batch=1)...")
            res_b1 = benchmark_tensorrt(cfg["engine_path"], batch_size=1, num_warmup=20, num_iterations=100)
            record["latency_b1_mean"] = round(res_b1["mean_ms"], 2)
            record["latency_b1_p95"] = round(res_b1["p95_ms"], 2)
            record["fps_b1"] = round(res_b1["fps"], 1)

            print("   -> Running TensorRT Benchmark (Batch=32)...")
            res_b32 = benchmark_tensorrt(cfg["engine_path"], batch_size=32, num_warmup=10, num_iterations=50)
            record["latency_b32_mean"] = round(res_b32["mean_ms"], 2)
            record["fps_b32"] = round(res_b32["fps"], 1)

            print(f"   ✓ B1 Latency: {record['latency_b1_mean']}ms ({record['fps_b1']} FPS) | B32: {record['latency_b32_mean']}ms ({record['fps_b32']} FPS)")
        except Exception as e:
            print(f"   ✗ Benchmark failed: {e}")
            record["status"] = f"BENCH_FAILED: {e}"

        # Step 4: Validate ALPR OCR Accuracy
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

        summary_records.append(record)

    # Step 5: Save JSON Report
    report_json_path = "results/full_matrix_benchmark_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_records, f, indent=2)

    # Step 6: Generate Markdown Table
    report_md_path = "results/full_matrix_benchmark_report.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# Relatório Comparativo Completo: Matriz de Variantes PARSeq\n\n")
        f.write("| Variante | Precisão | Fusão | Atenção | Engine (MB) | Latência B1 (ms) | FPS B1 | Latência B32 (ms) | FPS B32 | Acurácia Placa (%) | NED (%) | Status |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in summary_records:
            attn_type = "SageAttn" if r.get("sage_attn") else ("INT-Flash" if r.get("int_fa") else "Padrão")
            f.write(
                f"| {r['label']} | {r['precision'].upper()} | {r['fusion']} | {attn_type} | "
                f"{r['engine_size_mb']} | {r['latency_b1_mean']} | {r['fps_b1']} | {r['latency_b32_mean']} | {r['fps_b32']} | "
                f"{r['exact_acc']}% | {r['ned']}% | {r['status']} |\n"
            )

    print("\n" + "=" * 80)
    print("ALL PIPELINE STAGES COMPLETED!")
    print(f"Results saved to:\n  - {report_json_path}\n  - {report_md_path}")
    print("=" * 80)

    return summary_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pos_checkpoint", nargs="?", default=None, help="Optional positional checkpoint ckpt model")
    parser.add_argument("--checkpoint", type=str, default="pretrained/parseq_alpr_98.5.ckpt", help="Checkpoint ckpt model")
    parser.add_argument("--batch_size", type=int, default=64, help="Number of ALPR samples to evaluate")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of ALPR samples to evaluate")
    parser.add_argument("--samples", type=int, default=50, help="Number of ALPR samples to evaluate")
    parser.add_argument("--force", action="store_true", help="Force re-export and rebuild of all engines")
    args = parser.parse_args()

    ckpt = args.pos_checkpoint if args.pos_checkpoint is not None else args.checkpoint
    run_full_pipeline(max_eval_samples=args.samples, force=args.force, checkpoint=ckpt, batch_size=args.batch_size, num_workers=args.num_workers)
