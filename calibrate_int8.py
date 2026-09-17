#!/usr/bin/env python3
"""
calibrate_int8.py - End-to-End Post-Training Quantization (PTQ) for PARSeq
==========================================================================
Executes:
1. KL Divergence Histogram Calibration for Activations (Q8BERT / Bhandare et al.)
2. Per-channel Symmetric MinMax Calibration for Weights
3. IPTQ-ViT Unified Metric (Omega) search across all non-linear operators
4. Serializes calibrated checkpoint with static dyadic scales
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader, TensorDataset

from strhub.models.utils import create_model
from strhub.quantization.parseq_quantizer import PARSeqQuantizer
from strhub.quantization.unified_metric import UnifiedMetricSearcher


def parse_args():
    parser = argparse.ArgumentParser(description="PARSeq INT8 PTQ Calibration with Unified Metric")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to pretrained checkpoint or None for default")
    parser.add_argument("--data_root", type=str, default="data", help="Path to dataset root for real activation calibration")
    parser.add_argument("--num_batches", type=int, default=50, help="Number of calibration batches")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for calibration")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--output", type=str, default="parseq_int8_calibrated.pt", help="Path to save calibrated checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 65)
    print("PARSeq End-to-End INT8 PTQ Calibration Pipeline (IPTQ-ViT / HAWQ-V3)")
    print("=" * 65)
    print(f"Device: {args.device}")
    print(f"Calibration batches: {args.num_batches} (batch size {args.batch_size})")

    # Load base model preserving exact architecture hyperparameters
    from strhub.models.utils import load_from_checkpoint
    if args.checkpoint and os.path.isfile(args.checkpoint):
        print(f"[*] Loading checkpoint from {args.checkpoint}...")
        try:
            system = load_from_checkpoint(args.checkpoint)
            model = system.model
        except Exception:
            system = create_model("parseq", pretrained=False)
            ckpt = torch.load(args.checkpoint, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            clean_state = {k.replace("model.", ""): v for k, v in state_dict.items()}
            system.model.load_state_dict(clean_state, strict=False)
            model = system.model
    else:
        print("[*] Instantiating baseline PARSeq model...")
        system = create_model("parseq", pretrained=False)
        model = system.model

    model.to(args.device)
    model.eval()

    # Load real calibration data from dataset root
    if os.path.isdir(args.data_root):
        print(f"[*] Loading real calibration dataset from: {args.data_root}")
        from strhub.data.module import SceneTextDataModule
        hp = getattr(system, "hparams", None)
        img_size = hp.img_size if hp else (32, 128)
        max_label_len = hp.max_label_length if hp else 25
        charset_tr = hp.charset_train if hp else "0123456789abcdefghijklmnopqrstuvwxyz"
        charset_ts = hp.charset_test if hp else "0123456789abcdefghijklmnopqrstuvwxyz"
        dm = SceneTextDataModule(args.data_root, "_unused_", img_size, max_label_len, charset_tr, charset_ts, batch_size=args.batch_size, num_workers=2, augment=False)
        test_sets = SceneTextDataModule.TEST_BENCHMARK_SUB + SceneTextDataModule.TEST_BENCHMARK
        loaders = dm.test_dataloaders(test_sets)
        if loaders:
            dataloader = next(iter(loaders.values()))
        else:
            raise RuntimeError(f"No valid test subsets found in {args.data_root}")
    else:
        raise FileNotFoundError(f"Dataset root directory not found: '{args.data_root}'. Please provide a valid --data_root path.")

    # Initialize quantizer and prepare PTQ observers
    quantizer = PARSeqQuantizer(model, mode="ptq")
    ptq_model = quantizer.prepare_ptq()

    # Run calibration pass and Unified Metric search
    quantizer.calibrate(ptq_model, dataloader, num_batches=args.num_batches, device=args.device)

    # Build pure integer model
    print("[*] Constructing 100% Integer-Only PARSeq model with static dyadic scales...")
    integer_model = quantizer.build_integer_only()

    # Save calibrated model
    save_dict = {
        "state_dict": model.state_dict(),
        "integer_state_dict": integer_model.state_dict(),
        "assignments": quantizer.searcher.assignments,
        "scale_dict": quantizer.scale_dict,
        "hyper_parameters": dict(system.hparams) if hasattr(system, "hparams") else {},
    }
    torch.save(save_dict, args.output)
    print(f"[+] Successfully saved calibrated INT8 checkpoint to: {args.output}")
    print("=" * 65)


if __name__ == "__main__":
    main()
