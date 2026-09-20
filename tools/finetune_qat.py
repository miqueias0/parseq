import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, encoding="utf-8")

import time
import json
import argparse
from typing import Dict, Any, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from strhub.models.utils import load_from_checkpoint
from strhub.models.parseq.quantized_parseq import (
    create_model_variant,
    QuantizedLinear,
    load_calibration_into_model,
)
from strhub.data.module import SceneTextDataModule
from tools.evaluate_alpr import evaluate_dataset


def resolve_configuration(
    config_id: Optional[str] = None,
    variant: Optional[str] = None,
    fusion_level: Optional[str] = None,
    fuse_shapes: bool = False,
    fuse_mha: bool = False,
    fuse_mlp: bool = False,
    fuse_layernorm: bool = False,
    use_int_fa: bool = False,
    use_sage: bool = False,
    sage_mode: Optional[str] = None,
    v_quant_mode: Optional[str] = None,
    gelu_candidate: Optional[str] = None,
    softmax_candidate: Optional[str] = None,
    layernorm_candidate: Optional[str] = None,
    output_checkpoint: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolves configuration parameters from preset config_id and explicit user overrides."""
    cfg_match = None
    if config_id:
        try:
            from tools.run_full_matrix import CONFIGURATIONS
            cfg_match = next((c for c in CONFIGURATIONS if c["id"].lower() == config_id.lower()), None)
            if cfg_match is None:
                cfg_match = next((c for c in CONFIGURATIONS if config_id.lower() in c["id"].lower()), None)
        except ImportError:
            cfg_match = None

    if cfg_match:
        print(f"[Config Match] Matched preset ID '{cfg_match['id']}': {cfg_match['label']}")
        resolved_variant = variant if variant is not None else cfg_match.get("variant", "m6")
        resolved_fusion = fusion_level if fusion_level is not None else cfg_match.get("fusion_level", "none")
        resolved_int_fa = use_int_fa or cfg_match.get("use_int_fa", False)
        resolved_sage = use_sage or cfg_match.get("use_sage", False)
        resolved_sage_mode = sage_mode if sage_mode is not None else cfg_match.get("sage_mode", "sageattn_b")
        resolved_v_quant = v_quant_mode if v_quant_mode is not None else cfg_match.get("v_quant_mode", "per_tensor")
        resolved_gelu = gelu_candidate if gelu_candidate is not None else cfg_match.get("gelu_candidate", "gelu_iptq")
        resolved_sm = softmax_candidate if softmax_candidate is not None else cfg_match.get("softmax_candidate", "softmax_iptq")
        resolved_ln = layernorm_candidate if layernorm_candidate is not None else cfg_match.get("layernorm_candidate", "layernorm_ibert")
        if not output_checkpoint:
            output_checkpoint = f"pretrained/parseq_alpr_qat_{cfg_match['id']}.ckpt"
    else:
        if config_id:
            print(f"[Warning] config_id '{config_id}' not found in CONFIGURATIONS. Using passed parameters.")
        resolved_variant = variant or "m6"
        resolved_fusion = fusion_level or "none"
        resolved_int_fa = use_int_fa
        resolved_sage = use_sage
        resolved_sage_mode = sage_mode or "sageattn_b"
        resolved_v_quant = v_quant_mode or "per_tensor"
        resolved_gelu = gelu_candidate or "gelu_iptq"
        resolved_sm = softmax_candidate or "softmax_iptq"
        resolved_ln = layernorm_candidate or "layernorm_ibert"
        if not output_checkpoint:
            out_tag = config_id if config_id else f"{resolved_variant}_{resolved_fusion}"
            output_checkpoint = f"pretrained/parseq_alpr_qat_{out_tag}.ckpt"

    # Derive progressive kernel fusions
    resolved_fuse_shapes = fuse_shapes or (resolved_fusion in ["shapes", "mha", "mlp", "all"])
    resolved_fuse_mha = fuse_mha or (resolved_fusion in ["mha", "mlp", "all"])
    resolved_fuse_mlp = fuse_mlp or (resolved_fusion in ["mlp", "all"])
    resolved_fuse_ln = fuse_layernorm or (resolved_fusion in ["all"])

    return {
        "config_id": config_id or (cfg_match["id"] if cfg_match else None),
        "variant": resolved_variant,
        "fusion_level": resolved_fusion,
        "fuse_shapes": resolved_fuse_shapes,
        "fuse_mha": resolved_fuse_mha,
        "fuse_mlp": resolved_fuse_mlp,
        "fuse_layernorm": resolved_fuse_ln,
        "use_int_fa": resolved_int_fa,
        "use_sage": resolved_sage,
        "sage_mode": resolved_sage_mode,
        "v_quant_mode": resolved_v_quant,
        "gelu_candidate": resolved_gelu,
        "softmax_candidate": resolved_sm,
        "layernorm_candidate": resolved_ln,
        "output_checkpoint": output_checkpoint,
    }


def train_qat(
    checkpoint_path: str = "pretrained/parseq_alpr_98.5.ckpt",
    output_checkpoint: Optional[str] = None,
    dataset_name: str = "VeSV_pad",
    data_dir: str = "data",
    epochs: int = 3,
    lr: float = 1e-5,
    batch_size: int = 32,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    max_train_samples: int = 3000,
    val_samples: int = 200,
    seed: int = 42,
    device: str = "cuda",
    # Variant & Operator parameters
    config_id: Optional[str] = None,
    variant: Optional[str] = None,
    fusion_level: Optional[str] = None,
    fuse_shapes: bool = False,
    fuse_mha: bool = False,
    fuse_mlp: bool = False,
    fuse_layernorm: bool = False,
    use_int_fa: bool = False,
    use_sage: bool = False,
    sage_mode: Optional[str] = None,
    v_quant_mode: Optional[str] = None,
    gelu_candidate: Optional[str] = None,
    softmax_candidate: Optional[str] = None,
    layernorm_candidate: Optional[str] = None,
) -> Dict[str, Any]:
    # Resolve full configuration
    cfg = resolve_configuration(
        config_id=config_id,
        variant=variant,
        fusion_level=fusion_level,
        fuse_shapes=fuse_shapes,
        fuse_mha=fuse_mha,
        fuse_mlp=fuse_mlp,
        fuse_layernorm=fuse_layernorm,
        use_int_fa=use_int_fa,
        use_sage=use_sage,
        sage_mode=sage_mode,
        v_quant_mode=v_quant_mode,
        gelu_candidate=gelu_candidate,
        softmax_candidate=softmax_candidate,
        layernorm_candidate=layernorm_candidate,
        output_checkpoint=output_checkpoint,
    )

    out_ckpt = cfg["output_checkpoint"]

    # Set seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dev = torch.device(device if torch.cuda.is_available() and "cuda" in device else "cpu")
    print(f"\n================================================================================")
    print(f"STARTING FINE-TUNING / QAT ON {dev}")
    print(f"================================================================================")
    print(f"Config ID:         {cfg['config_id'] or 'Custom Manual Configuration'}")
    print(f"Variant:           {cfg['variant'].upper()} (Quantization Mode)")
    print(f"Base Checkpoint:   {checkpoint_path}")
    print(f"Output Checkpoint: {out_ckpt}")
    print(f"Fusion Level:      {cfg['fusion_level']} (shapes={cfg['fuse_shapes']}, mha={cfg['fuse_mha']}, mlp={cfg['fuse_mlp']}, ln={cfg['fuse_layernorm']})")
    print(f"Attention Type:    INT-FA={cfg['use_int_fa']}, SageAttn={cfg['use_sage']} (mode={cfg['sage_mode']}, v_quant={cfg['v_quant_mode']})")
    print(f"Approximations:    GELU={cfg['gelu_candidate']}, Softmax={cfg['softmax_candidate']}, LN={cfg['layernorm_candidate']}")
    print(f"Hyperparameters:   epochs={epochs}, lr={lr}, batch_size={batch_size}, weight_decay={weight_decay}, grad_clip={grad_clip}")
    print(f"================================================================================\n")

    # Load base model
    system = load_from_checkpoint(checkpoint_path)
    hp = system.hparams

    # Locate calibration stats file
    calib_candidates = [
        "calibration_stats.json",
        "results/calibration_stats.json",
        "results/calibration/calibration_stats.json",
    ]
    calib_file = next((c for c in calib_candidates if os.path.exists(c)), None)
    if calib_file:
        print(f"Found calibration file: {calib_file}")

    # Build model variant (use_plugin=False ensures PyTorch autograd graph is intact)
    model = create_model_variant(
        variant=cfg["variant"],
        base_model=system.model,
        gelu_candidate=cfg["gelu_candidate"],
        softmax_candidate=cfg["softmax_candidate"],
        layernorm_candidate=cfg["layernorm_candidate"],
        calibration_file=calib_file,
        fuse_mha=cfg["fuse_mha"],
        fuse_mlp=cfg["fuse_mlp"],
        fuse_layernorm=cfg["fuse_layernorm"],
        use_int_flashattention=cfg["use_int_fa"],
        use_sage_attention=cfg["use_sage"],
        sage_mode=cfg["sage_mode"],
        v_quant_mode=cfg["v_quant_mode"],
        use_plugin=False,
    )

    # Load existing fine-tuned weights if checkpoint_path is already a trained checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path) and os.path.normpath(checkpoint_path) != os.path.normpath("pretrained/parseq_alpr_98.5.ckpt"):
        try:
            ckpt_data = torch.load(checkpoint_path, map_location=dev, weights_only=False)
            sd = ckpt_data.get("model_state_dict", ckpt_data.get("state_dict", ckpt_data))
            if sd:
                clean_sd = {k.replace("model.", ""): v for k, v in sd.items()}
                model.load_state_dict(clean_sd, strict=False)
                for m in model.modules():
                    if isinstance(m, QuantizedLinear):
                        m.recompute_weight_scale()
                print(f"Loaded existing weights from {checkpoint_path}")
        except Exception as e:
            print(f"Notice: Could not load initial state_dict from {checkpoint_path}: {e}")

    model.to(dev)

    # Ensure trainable parameters exist with Straight-Through Estimator (STE)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        print("Notice: Enabling grad and STE on QuantizedLinear weights for fine-tuning...")
        for m in model.modules():
            if isinstance(m, QuantizedLinear):
                m.mode = "qat"
                m.weight.requires_grad_(True)
                if m.bias is not None:
                    m.bias.requires_grad_(True)
        trainable_params = [p for p in model.parameters() if p.requires_grad]

    print(f"Number of trainable tensors: {len(trainable_params)} ({sum(p.numel() for p in trainable_params):,} parameters)")

    # Data module
    dm = SceneTextDataModule(
        root_dir=data_dir,
        train_dir=dataset_name,
        img_size=hp.img_size,
        max_label_length=hp.max_label_length,
        charset_train=hp.charset_train,
        charset_test=hp.charset_test,
        batch_size=batch_size,
        num_workers=0,
        augment=False
    )

    train_data = dm.train_dataset
    if max_train_samples and max_train_samples < len(train_data):
        indices = list(range(max_train_samples))
        train_subset = Subset(train_data, indices)
    else:
        train_subset = train_data

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = dm.test_dataloaders([dataset_name])[dataset_name]

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    total_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1), eta_min=1e-7)

    history = {
        "hyperparameters": {
            "config_id": cfg["config_id"],
            "variant": cfg["variant"],
            "fusion_level": cfg["fusion_level"],
            "fuse_shapes": cfg["fuse_shapes"],
            "fuse_mha": cfg["fuse_mha"],
            "fuse_mlp": cfg["fuse_mlp"],
            "fuse_layernorm": cfg["fuse_layernorm"],
            "use_int_fa": cfg["use_int_fa"],
            "use_sage": cfg["use_sage"],
            "sage_mode": cfg["sage_mode"],
            "v_quant_mode": cfg["v_quant_mode"],
            "gelu_candidate": cfg["gelu_candidate"],
            "softmax_candidate": cfg["softmax_candidate"],
            "layernorm_candidate": cfg["layernorm_candidate"],
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "grad_clip": grad_clip,
            "max_train_samples": len(train_subset),
            "seed": seed,
            "base_checkpoint": checkpoint_path,
            "output_checkpoint": out_ckpt,
        },
        "epochs": []
    }

    # Initial pre-QAT validation
    print("\nEvaluating initial pre-training performance on validation set...")
    val_init = evaluate_dataset(
        model=model,
        data_loader=val_loader,
        tokenizer=system.tokenizer,
        charset_adapter=system.charset_adapter,
        device=dev,
        max_samples=val_samples
    )
    print(f"Pre-Training Val Accuracy: {val_init['exact_plate_accuracy']*100:.2f}%, CER: {val_init['character_error_rate']*100:.2f}%")

    history["pre_training_metrics"] = {
        "exact_plate_acc": val_init["exact_plate_accuracy"] * 100.0,
        "cer": val_init["character_error_rate"] * 100.0,
        "ned": val_init["normalized_edit_distance"] * 100.0
    }

    best_acc = val_init["exact_plate_accuracy"]

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        step_count = 0
        t0 = time.time()

        for step, (images, labels) in enumerate(train_loader):
            images = images.to(dev)
            targets = system.tokenizer.encode(labels, device=images.device)
            # targets contains [<bos>, char1, char2, ..., <eos>]
            targets_target = targets[:, 1:]
            
            optimizer.zero_grad()
            logits = model(system.tokenizer, images)
            
            # Match lengths safely
            min_len = min(logits.shape[1], targets_target.shape[1])
            logits_slice = logits[:, :min_len]
            targets_slice = targets_target[:, :min_len]

            loss = F.cross_entropy(
                logits_slice.reshape(-1, logits.shape[-1]),
                targets_slice.reshape(-1),
                ignore_index=system.tokenizer.pad_id
            )

            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=grad_clip)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            step_count += 1

            if (step + 1) % 25 == 0 or (step + 1) == len(train_loader):
                print(f"Epoch [{epoch}/{epochs}] Step [{step+1}/{len(train_loader)}] Loss: {loss.item():.4f} LR: {scheduler.get_last_lr()[0]:.2e}")

        epoch_loss = running_loss / max(step_count, 1)
        epoch_time = time.time() - t0

        # Validation
        model.eval()
        val_res = evaluate_dataset(
            model=model,
            data_loader=val_loader,
            tokenizer=system.tokenizer,
            charset_adapter=system.charset_adapter,
            device=dev,
            max_samples=val_samples
        )

        val_acc = val_res["exact_plate_accuracy"] * 100.0
        val_cer = val_res["character_error_rate"] * 100.0
        val_ned = val_res["normalized_edit_distance"] * 100.0

        print(f"--> Epoch {epoch} Finished in {epoch_time:.1f}s | Train Loss: {epoch_loss:.4f} | Val Plate Acc: {val_acc:.2f}% | Val CER: {val_cer:.2f}% | Val NED: {val_ned:.2f}%")

        epoch_stats = {
            "epoch": epoch,
            "train_loss": epoch_loss,
            "val_exact_plate_acc": val_acc,
            "val_cer": val_cer,
            "val_ned": val_ned,
            "time_seconds": epoch_time
        }
        history["epochs"].append(epoch_stats)

        # Save best model
        if val_res["exact_plate_accuracy"] >= best_acc:
            best_acc = val_res["exact_plate_accuracy"]
            os.makedirs(os.path.dirname(os.path.abspath(out_ckpt)), exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "hparams": hp,
                "history": history,
                "config_id": cfg["config_id"],
                "variant": cfg["variant"],
                "fusion_level": cfg["fusion_level"],
                "fuse_shapes": cfg["fuse_shapes"],
                "fuse_mha": cfg["fuse_mha"],
                "fuse_mlp": cfg["fuse_mlp"],
                "fuse_layernorm": cfg["fuse_layernorm"],
                "use_int_fa": cfg["use_int_fa"],
                "use_sage": cfg["use_sage"],
                "sage_mode": cfg["sage_mode"],
                "v_quant_mode": cfg["v_quant_mode"],
                "gelu_candidate": cfg["gelu_candidate"],
                "softmax_candidate": cfg["softmax_candidate"],
                "layernorm_candidate": cfg["layernorm_candidate"],
            }, out_ckpt)
            print(f"Saved new best checkpoint to {out_ckpt}")

    # Save history json
    os.makedirs("results", exist_ok=True)
    hist_name = f"qat_training_{cfg['config_id']}_history.json" if cfg["config_id"] else "qat_training_history.json"
    hist_path = os.path.join("results", hist_name)
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {hist_path}")

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune / QAT PARSeq model for specific evaluation variants.")
    parser.add_argument("--config_id", type=str, default=None, help="Preset configuration ID from tools.run_full_matrix (e.g. m6_int_fa_fuse_all, m6_fused_all, m6_sage_b_tensor_fuse_all)")
    parser.add_argument("--variant", type=str, default=None, help="Model variant (default: m6 or from config_id)")
    parser.add_argument("--checkpoint", type=str, default="pretrained/parseq_alpr_98.5.ckpt", help="Initial checkpoint path")
    parser.add_argument("--output", type=str, default=None, help="Destination checkpoint path (defaults to pretrained/parseq_alpr_qat_{config_id}.ckpt)")
    parser.add_argument("--dataset", type=str, default="VeSV_pad", help="Dataset name in data_dir")
    parser.add_argument("--data_dir", type=str, default="data", help="Root data directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping max norm")
    parser.add_argument("--max_train_samples", type=int, default=2000, help="Max train samples per epoch")
    parser.add_argument("--val_samples", type=int, default=200, help="Validation samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Target device")

    # Fusion flags
    parser.add_argument("--fusion_level", type=str, default=None, choices=["none", "shapes", "mha", "mlp", "all"], help="Kernel fusion level")
    parser.add_argument("--fuse_shapes", action="store_true", help="Fuse reshape/transpose ops")
    parser.add_argument("--fuse_mha", action="store_true", help="Fuse MHA into FMHA pattern")
    parser.add_argument("--fuse_mlp", action="store_true", help="Fuse MLP activation (FP32 GELU)")
    parser.add_argument("--fuse_layernorm", action="store_true", help="Fuse LayerNorm (FP32 LayerNorm)")

    # Attention flags
    parser.add_argument("--use_int_fa", "--use_int_flashattention", dest="use_int_fa", action="store_true", help="Use INT-FlashAttention")
    parser.add_argument("--use_sage", "--use_sage_attention", dest="use_sage", action="store_true", help="Use SageAttention")
    parser.add_argument("--sage_mode", type=str, default=None, choices=["sageattn_a", "sageattn_b", "sageattn_vb"], help="SageAttention operational mode")
    parser.add_argument("--v_quant_mode", type=str, default=None, choices=["per_tensor", "per_head", "per_channel_int8", "per_channel_fp8"], help="Value quantization mode")

    # Non-linear candidate operators
    parser.add_argument("--gelu_candidate", type=str, default=None, choices=["gelu_iptq", "gelu_poly_deg2", "gelu_lut_fp16", "gelu_fp32", "gelu_ibert", "gelu_ivit"], help="GELU approximation candidate")
    parser.add_argument("--softmax_candidate", type=str, default=None, choices=["softmax_iptq", "softmax_lut_fp16", "softmax_taylor_deg2", "softmax_fp32", "softmax_ibert", "softmax_ivit"], help="Softmax approximation candidate")
    parser.add_argument("--layernorm_candidate", type=str, default=None, choices=["layernorm_ibert", "layernorm_power_of_two", "layernorm_lut", "layernorm_iptq", "layernorm_fp32"], help="LayerNorm approximation candidate")

    args = parser.parse_args()

    train_qat(
        checkpoint_path=args.checkpoint,
        output_checkpoint=args.output,
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        max_train_samples=args.max_train_samples,
        val_samples=args.val_samples,
        seed=args.seed,
        device=args.device,
        config_id=args.config_id,
        variant=args.variant,
        fusion_level=args.fusion_level,
        fuse_shapes=args.fuse_shapes,
        fuse_mha=args.fuse_mha,
        fuse_mlp=args.fuse_mlp,
        fuse_layernorm=args.fuse_layernorm,
        use_int_fa=args.use_int_fa,
        use_sage=args.use_sage,
        sage_mode=args.sage_mode,
        v_quant_mode=args.v_quant_mode,
        gelu_candidate=args.gelu_candidate,
        softmax_candidate=args.softmax_candidate,
        layernorm_candidate=args.layernorm_candidate,
    )
