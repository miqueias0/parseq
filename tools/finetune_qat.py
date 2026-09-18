import os
import sys
import time
import json
import argparse
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from strhub.models.utils import load_from_checkpoint
from strhub.models.parseq.quantized_parseq import create_model_variant
from strhub.data.module import SceneTextDataModule
from tools.evaluate_alpr import evaluate_dataset


def train_qat(
    checkpoint_path: str = "pretrained/parseq_alpr_98.5.ckpt",
    output_checkpoint: str = "pretrained/parseq_alpr_qat_m6.ckpt",
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
    device: str = "cuda"
) -> Dict[str, Any]:
    # Set seeds (Section 76)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dev = torch.device(device if torch.cuda.is_available() and "cuda" in device else "cpu")
    print(f"=== Starting Quantization-Aware Training (QAT - M6) on {dev} ===")
    print(f"Parameters: epochs={epochs}, lr={lr}, batch_size={batch_size}, weight_decay={weight_decay}, grad_clip={grad_clip}")

    system = load_from_checkpoint(checkpoint_path)
    hp = system.hparams

    # Build M6 variant with STE in QuantizedLinear
    model = create_model_variant("m6", system.model)
    model.to(dev)

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

    # Optimizer & Scheduler (Section 34)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    total_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1), eta_min=1e-7)

    history = {
        "hyperparameters": {
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "grad_clip": grad_clip,
            "max_train_samples": len(train_subset),
            "seed": seed,
            "base_checkpoint": checkpoint_path
        },
        "epochs": []
    }

    # Initial pre-QAT validation
    print("\nEvaluating initial pre-QAT M6 performance on validation set...")
    val_init = evaluate_dataset(
        model=model,
        data_loader=val_loader,
        tokenizer=system.tokenizer,
        charset_adapter=system.charset_adapter,
        device=dev,
        max_samples=val_samples
    )
    print(f"Pre-QAT Val Accuracy: {val_init['exact_plate_accuracy']*100:.2f}%, CER: {val_init['character_error_rate']*100:.2f}%")

    history["pre_qat_metrics"] = {
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
            L_tgt = targets_target.shape[1]

            optimizer.zero_grad()
            logits = model(system.tokenizer, images)
            logits_slice = logits[:, :L_tgt]

            loss = F.cross_entropy(
                logits_slice.reshape(-1, logits.shape[-1]),
                targets_target.reshape(-1),
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

        # Validation (Section 33)
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
            os.makedirs(os.path.dirname(os.path.abspath(output_checkpoint)), exist_ok=True)
            # Save state dict
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "hparams": hp,
                "history": history
            }, output_checkpoint)
            print(f"Saved new best QAT checkpoint to {output_checkpoint}")

    # Save history json (Section 33)
    os.makedirs("results", exist_ok=True)
    with open("results/qat_training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print("QAT training history saved to results/qat_training_history.json")

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="pretrained/parseq_alpr_98.5.ckpt")
    parser.add_argument("--output", type=str, default="pretrained/parseq_alpr_qat_m6.ckpt")
    parser.add_argument("--dataset", type=str, default="VeSV_pad")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_train_samples", type=int, default=2000)
    parser.add_argument("--val_samples", type=int, default=200)
    args = parser.parse_args()

    train_qat(
        checkpoint_path=args.checkpoint,
        output_checkpoint=args.output,
        dataset_name=args.dataset,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        max_train_samples=args.max_train_samples,
        val_samples=args.val_samples
    )
