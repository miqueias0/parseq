import os
import sys
import json
import argparse
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

# Add workspace root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import torch
from torch.utils.data import DataLoader
from nltk import edit_distance

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint
from strhub.models.parseq.quantized_parseq import create_model_variant


class ONNXModelWrapper(torch.nn.Module):
    """Wrapper exposing standard PyTorch forward(images) -> logits via ONNX Runtime."""
    def __init__(self, onnx_path: str, device: str = "cuda"):
        super().__init__()
        import onnxruntime as ort
        self.device = torch.device(device if torch.cuda.is_available() and "cuda" in device else "cpu")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "cuda" in str(self.device) and "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
        print(f"Loading ONNX Model for ALPR Evaluation: {onnx_path} using {providers[0]}")
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def forward(self, *args, **kwargs) -> torch.Tensor:
        images = args[-1] if len(args) > 0 and isinstance(args[-1], torch.Tensor) else None
        if images is None:
            for v in list(args) + list(kwargs.values()):
                if isinstance(v, torch.Tensor):
                    images = v
                    break
        bs = images.shape[0]
        logits_list = []
        for i in range(bs):
            img_single = images[i:i+1].cpu().numpy()
            out_np = self.session.run([self.output_name], {self.input_name: img_single})[0]
            logits_list.append(torch.from_numpy(out_np))
        logits = torch.cat(logits_list, dim=0)
        return logits.to(device=images.device)


class TensorRTModelWrapper(torch.nn.Module):
    """Wrapper exposing standard PyTorch forward(images) -> logits via TensorRT engine with dedicated stream."""
    def __init__(self, engine_path: str, device: str = "cuda"):
        super().__init__()
        import tensorrt as trt
        from strhub.quant.plugins.trt_plugins import register_parseq_plugins
        register_parseq_plugins()
        self.device = torch.device("cuda")
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)
        print(f"Loading TensorRT Engine for ALPR Evaluation: {engine_path}")
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()

        in_dtype = self.engine.get_tensor_dtype("images")
        self.torch_in_dtype = torch.float16 if in_dtype == trt.DataType.HALF else torch.float32
        out_dtype = self.engine.get_tensor_dtype("logits")
        self.torch_out_dtype = torch.float16 if out_dtype == trt.DataType.HALF else torch.float32

        # Preallocate static slice buffers to eliminate per-slice GPU allocations
        self.d_single_in = torch.empty((1, 3, 32, 128), dtype=self.torch_in_dtype, device="cuda")
        self.context.set_input_shape("images", (1, 3, 32, 128))
        out_shape = tuple(self.context.get_tensor_shape("logits"))
        self.d_single_out = torch.empty(out_shape, dtype=self.torch_out_dtype, device="cuda")
        self.out_len = out_shape[1]
        self.num_classes = out_shape[2]
        self.context.set_tensor_address("images", int(self.d_single_in.data_ptr()))
        self.context.set_tensor_address("logits", int(self.d_single_out.data_ptr()))

    def forward(self, *args, **kwargs) -> torch.Tensor:
        images = args[-1] if len(args) > 0 and isinstance(args[-1], torch.Tensor) else None
        if images is None:
            for v in list(args) + list(kwargs.values()):
                if isinstance(v, torch.Tensor):
                    images = v
                    break
        bs = images.shape[0]

        # Transfer full batch to GPU at once
        d_batch_imgs = images.to(device="cuda", dtype=self.torch_in_dtype, non_blocking=True)
        batch_logits = torch.empty((bs, self.out_len, self.num_classes), dtype=torch.float32, device="cuda")

        # Pipelined execution without CPU blocking per sample
        with torch.cuda.stream(self.stream):
            for i in range(bs):
                self.d_single_in.copy_(d_batch_imgs[i:i+1], non_blocking=True)
                self.context.execute_async_v3(self.stream.cuda_stream)
                batch_logits[i:i+1].copy_(self.d_single_out, non_blocking=True)

        self.stream.synchronize()
        return batch_logits.to(device=images.device)


def compute_bootstrap_ci(data: List[float], n_bootstrap: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
    """Compute non-parametric bootstrap confidence interval."""
    if not data:
        return 0.0, 0.0
    arr = np.array(data)
    boot_means = []
    n = len(arr)
    rng = np.random.default_rng(seed=42)
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means.append(np.mean(sample))
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, alpha * 100))
    upper = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    return lower, upper


def evaluate_dataset(
    model: torch.nn.Module,
    data_loader: DataLoader,
    tokenizer,
    charset_adapter,
    device: torch.device,
    max_samples: Optional[int] = None
) -> Dict[str, Any]:
    model.eval()
    model.to(device)

    exact_matches = []
    ned_scores = []
    cer_scores = []
    confidences = []
    length_matches = defaultdict(list)
    char_accuracies = defaultdict(lambda: {"correct": 0, "total": 0})
    confusion_pairs = defaultdict(int)
    error_types = {"substitution": 0, "deletion": 0, "insertion": 0, "length_mismatch": 0}

    total_evaluated = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(data_loader):
            if max_samples and total_evaluated >= max_samples:
                break

            model_dtype = next(model.parameters()).dtype if list(model.parameters()) else torch.float32
            images = images.to(device=device, dtype=model_dtype)
            # Forward pass
            if hasattr(model, "tokenizer"):
                logits = model(images)
            else:
                try:
                    logits = model(tokenizer, images)
                except TypeError:
                    logits = model(images)
            probs = logits.softmax(-1)
            preds, prob_tuples = tokenizer.decode(probs)

            for pred, prob, gt in zip(preds, prob_tuples, labels):
                pred = charset_adapter(pred)
                conf = float(prob.prod().item()) if hasattr(prob, "prod") else 1.0
                confidences.append(conf)

                is_exact = 1.0 if pred == gt else 0.0
                exact_matches.append(is_exact)

                # ICDAR definition of NED
                ed = edit_distance(pred, gt)
                max_l = max(len(pred), len(gt), 1)
                ned = 1.0 - (ed / max_l)
                ned_scores.append(ned)

                cer = ed / max(len(gt), 1)
                cer_scores.append(cer)

                # Length breakdown
                length_matches[len(gt)].append(is_exact)

                # Character-level statistics & confusion
                for c_gt in gt:
                    char_accuracies[c_gt]["total"] += 1

                if is_exact:
                    for c_gt in gt:
                        char_accuracies[c_gt]["correct"] += 1
                else:
                    if len(pred) != len(gt):
                        error_types["length_mismatch"] += 1
                        if len(pred) < len(gt):
                            error_types["deletion"] += (len(gt) - len(pred))
                        else:
                            error_types["insertion"] += (len(pred) - len(gt))

                    # Track common character confusions when lengths match
                    if len(pred) == len(gt):
                        for c_p, c_g in zip(pred, gt):
                            if c_p == c_g:
                                char_accuracies[c_g]["correct"] += 1
                            else:
                                error_types["substitution"] += 1
                                pair = f"{c_g}->{c_p}"
                                confusion_pairs[pair] += 1

                total_evaluated += 1
                if max_samples and total_evaluated >= max_samples:
                    break

    exact_acc = float(np.mean(exact_matches)) if exact_matches else 0.0
    mean_ned = float(np.mean(ned_scores)) if ned_scores else 0.0
    mean_cer = float(np.mean(cer_scores)) if cer_scores else 0.0
    mean_conf = float(np.mean(confidences)) if confidences else 0.0

    ci_exact_low, ci_exact_high = compute_bootstrap_ci(exact_matches)
    ci_ned_low, ci_ned_high = compute_bootstrap_ci(ned_scores)

    # Per-character accuracy
    per_char_acc = {}
    for char, counts in sorted(char_accuracies.items()):
        acc = counts["correct"] / max(counts["total"], 1)
        per_char_acc[char] = {"accuracy": float(acc), "total": counts["total"]}

    # Top confusion pairs
    top_confusions = dict(sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)[:15])

    # Per-length accuracy
    per_length_acc = {str(k): float(np.mean(v)) for k, v in sorted(length_matches.items())}

    return {
        "num_samples": total_evaluated,
        "exact_plate_accuracy": exact_acc,
        "exact_plate_accuracy_ci95": [ci_exact_low, ci_exact_high],
        "normalized_edit_distance": mean_ned,
        "normalized_edit_distance_ci95": [ci_ned_low, ci_ned_high],
        "character_error_rate": mean_cer,
        "plate_error_rate": 1.0 - exact_acc,
        "mean_confidence": mean_conf,
        "per_length_accuracy": per_length_acc,
        "error_types": error_types,
        "top_confusion_pairs": top_confusions,
        "per_character_accuracy": per_char_acc,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="pretrained/parseq_alpr_98.5.ckpt",
                        help="Model checkpoint (.ckpt) or path to (.onnx) or (.engine)")
    parser.add_argument("--model", type=str, default=None,
                        help="Optional explicit path to .onnx, .engine, or .ckpt model")
    parser.add_argument("--variant", type=str, default="m1", choices=["m0", "m1", "m2", "m3", "m4", "m5", "m6"],
                        help="Architecture/quantization variant when evaluating from .ckpt")
    parser.add_argument("--base_checkpoint", type=str, default="pretrained/parseq_alpr_98.5.ckpt",
                        help="Base checkpoint providing tokenizer and metadata when testing .onnx or .engine")
    parser.add_argument("--dataset", type=str, default="VeSV_pad")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_int_flashattention", action="store_true", default=False,
                        help="Habilita INT-FlashAttention (arXiv:2409.16997v2) com GEMMs INT8 e online softmax fundido")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")

    # Determine target model file
    target_path = args.model if args.model else args.checkpoint
    target_lower = target_path.lower()

    # Load base system for tokenizer and charset_adapter
    system = load_from_checkpoint(args.base_checkpoint).eval()
    hp = system.hparams

    if target_lower.endswith(".onnx"):
        print(f"--- Evaluating ONNX Model: {target_path} on {args.dataset} ---")
        model = ONNXModelWrapper(target_path, device=args.device)
    elif target_lower.endswith(".engine"):
        print(f"--- Evaluating TensorRT Engine: {target_path} on {args.dataset} ---")
        model = TensorRTModelWrapper(target_path, device=args.device)
    else:
        # PyTorch checkpoint evaluation
        if "qat" in target_lower or args.variant == "m6":
            system = load_from_checkpoint(args.base_checkpoint).eval()
            model = create_model_variant(
                "m6", system.model, use_int_flashattention=args.use_int_flashattention
            ).eval().to(dev)
            qat_ckpt_path = target_path if "qat" in target_lower else "pretrained/parseq_alpr_qat_m6.ckpt"
            if os.path.exists(qat_ckpt_path):
                ckpt = torch.load(qat_ckpt_path, map_location=dev, weights_only=False)
                sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
                clean_sd = {k.replace("model.", ""): v for k, v in sd.items()}
                model.load_state_dict(clean_sd, strict=False)
                print(f"Loaded trained QAT checkpoint from {qat_ckpt_path} into variant M6.")
        else:
            system = load_from_checkpoint(target_path).eval()
            model = create_model_variant(
                args.variant, system.model, use_int_flashattention=args.use_int_flashattention
            ).eval().to(dev)

        print(f"--- Evaluating PyTorch Variant {args.variant.upper()} on {args.dataset} ---")

    datamodule = SceneTextDataModule(
        root_dir=args.data_root,
        train_dir="_unused_",
        img_size=hp.img_size,
        max_label_length=hp.max_label_length,
        charset_train=hp.charset_train,
        charset_test=hp.charset_test,
        batch_size=args.batch_size,
        num_workers=0,
        augment=False
    )
    test_loaders = datamodule.test_dataloaders([args.dataset])
    loader = test_loaders[args.dataset]

    metrics = evaluate_dataset(
        model=model,
        data_loader=loader,
        tokenizer=system.tokenizer,
        charset_adapter=system.charset_adapter,
        device=dev,
        max_samples=args.max_samples
    )

    print(f"Evaluated Samples: {metrics['num_samples']}")
    print(f"Exact Plate Accuracy: {metrics['exact_plate_accuracy'] * 100:.2f}% (95% CI: [{metrics['exact_plate_accuracy_ci95'][0]*100:.2f}%, {metrics['exact_plate_accuracy_ci95'][1]*100:.2f}%])")
    print(f"Normalized Edit Distance (NED): {metrics['normalized_edit_distance'] * 100:.2f}%")
    print(f"Character Error Rate (CER): {metrics['character_error_rate'] * 100:.2f}%")
    print(f"Plate Error Rate: {metrics['plate_error_rate'] * 100:.2f}%")
    print(f"Top Confusions: {metrics['top_confusion_pairs']}")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved results to {args.output}")

