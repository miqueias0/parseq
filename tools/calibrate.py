import os
import sys
import json
import argparse
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint
from strhub.models.parseq.quantized_parseq import create_model_variant
from strhub.quant.quant_utils import (
    quantize_symmetric,
    compute_saturation_stats,
    compute_distribution_stats,
)


class CalibrationHook:
    def __init__(self, name: str):
        self.name = name
        self.activations = []

    def hook(self, module, input, output):
        # Activation quantization applies to the input tensor entering the linear/norm operation
        act = input[0] if (isinstance(input, (tuple, list)) and len(input) > 0) else input
        if isinstance(act, torch.Tensor):
            self.activations.append(act.detach().cpu())


def run_calibration(
    checkpoint_path: str,
    dataset_name: str = "VeSV_pad",
    data_dir: str = "data",
    samples_list: List[int] = [32, 64, 128, 256, 512, 1000],
    output_dir: str = "results/calibration"
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    # Load base model
    system = load_from_checkpoint(checkpoint_path)
    base_model = system.model
    base_model.eval()
    base_model.to(device)

    # Load calibration data strictly from train split (preventing data leakage, Section 35)
    data_module = SceneTextDataModule(
        root_dir=data_dir,
        train_dir=dataset_name,
        img_size=system.hparams.img_size,
        max_label_length=system.hparams.max_label_length,
        charset_train=system.hparams.charset_train,
        charset_test=system.hparams.charset_test,
        batch_size=32,
        num_workers=0,
        augment=False
    )
    train_dataset = data_module.train_dataset

    results_by_sample_size = {}

    max_requested = max(samples_list)
    max_n = min(max_requested, len(train_dataset))
    print(f"--> Collecting activations across {max_n} calibration samples (single pass)...", flush=True)

    indices = list(range(max_n))
    sub_dataset = Subset(train_dataset, indices)
    loader = DataLoader(sub_dataset, batch_size=32, shuffle=False)

    # Register hooks on encoder blocks and decoder
    hooks = {}
    handles = []
    for name, module in base_model.named_modules():
        if isinstance(module, (nn.Linear, nn.LayerNorm)):
            hook_obj = CalibrationHook(name)
            hooks[name] = hook_obj
            handles.append(module.register_forward_hook(hook_obj.hook))

    # Single forward pass
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            if hasattr(base_model, "tokenizer"):
                _ = base_model(images)
            else:
                _ = base_model(system.tokenizer, images)

    for h in handles:
        h.remove()

    print("--> Forward collection completed. Computing distribution and saturation statistics across sample sizes...", flush=True)

    results_by_sample_size = {}
    for n_samples in samples_list:
        cur_n = min(n_samples, max_n)
        layer_stats = {}
        for name, hook_obj in hooks.items():
            if not hook_obj.activations:
                continue
            # Collect activations for cur_n samples
            # Each batch has 32 samples (or remainder)
            needed_batches = (cur_n + 31) // 32
            sliced_acts = hook_obj.activations[:needed_batches]
            flat_act = torch.cat([a.flatten() for a in sliced_acts], dim=0)

            dist_stats = compute_distribution_stats(flat_act)
            q, scale = quantize_symmetric(flat_act, bits=8)
            sat_stats = compute_saturation_stats(q, bits=8)

            stat_entry = {
                "scale": float(scale.item()) if scale.numel() == 1 else float(scale.mean().item()),
                "saturation": sat_stats,
                "distribution": dist_stats,
            }
            layer_stats[name] = stat_entry
            # Also store alias for wrapped attention modules
            if ".attn." in name and ".attn.attn." not in name:
                layer_stats[name.replace(".attn.", ".attn.attn.")] = stat_entry

        results_by_sample_size[str(n_samples)] = {
            "samples": cur_n,
            "layers": layer_stats
        }
        print(f"  Processed sample size {cur_n}.", flush=True)

    # Save to JSON
    out_path = os.path.join(output_dir, "calibration_stats.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_by_sample_size, f, indent=2)

    return results_by_sample_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="pretrained/parseq_alpr_98.5.ckpt")
    parser.add_argument("--dataset", type=str, default="VeSV_pad")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="results/calibration")
    args = parser.parse_args()

    stats = run_calibration(
        checkpoint_path=args.checkpoint,
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    print(f"Calibration completed for {len(stats)} sample sizes. Saved to {args.output_dir}.")
