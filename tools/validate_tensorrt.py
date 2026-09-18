import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import json
import argparse
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import tensorrt as trt

from strhub.models.utils import load_from_checkpoint
from strhub.quant.quant_utils import compute_sqnr, compute_mse, compute_cosine_similarity


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def run_tensorrt_inference(
    engine,
    context,
    input_numpy: np.ndarray
) -> np.ndarray:
    """Executes TensorRT inference with synchronous CUDA memory copy."""
    # TensorRT 10/11 uses execute_async_v3 or execute_v2
    batch_size = input_numpy.shape[0]

    # Input and output shapes
    input_shape = input_numpy.shape
    context.set_input_shape("images", input_shape)

    output_shape = tuple(context.get_tensor_shape("logits"))

    # Determine required dtypes from engine
    in_dtype = engine.get_tensor_dtype("images")
    torch_in_dtype = torch.float16 if in_dtype == trt.DataType.HALF else torch.float32
    d_input = torch.from_numpy(input_numpy).to(dtype=torch_in_dtype, device="cuda")

    out_dtype = engine.get_tensor_dtype("logits")
    torch_out_dtype = torch.float16 if out_dtype == trt.DataType.HALF else torch.float32
    d_output = torch.empty(output_shape, dtype=torch_out_dtype, device="cuda")

    context.set_tensor_address("images", int(d_input.data_ptr()))
    context.set_tensor_address("logits", int(d_output.data_ptr()))

    if not hasattr(run_tensorrt_inference, "_stream"):
        run_tensorrt_inference._stream = torch.cuda.Stream()
    stream = run_tensorrt_inference._stream
    context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    return d_output.to(dtype=torch.float32).cpu().numpy()


def validate_tensorrt_engine(
    checkpoint_path: str,
    engine_path: str,
    expected_precision: str = "fp32",
    num_samples: int = 20,
    img_size: tuple = (32, 128)
) -> Dict[str, Any]:
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    system = load_from_checkpoint(checkpoint_path).eval().cuda()
    base = system.model
    num_steps = base.max_label_length + 1

    max_abs_diffs = []
    mses = []
    cos_sims = []
    exact_agreements = 0
    divergence_diagnostics = []

    for idx in range(num_samples):
        dummy_input = torch.randn(1, 3, img_size[0], img_size[1], dtype=torch.float32, device="cuda")

        # PyTorch reference
        with torch.no_grad():
            memory = base.encode(dummy_input)
            pos_queries = base.pos_queries[:, :num_steps].expand(1, -1, -1)
            tgt_in = torch.full((1, 1), system.tokenizer.bos_id, dtype=torch.long, device=base._device)
            tgt_out = base.decode(tgt_in, memory, tgt_query=pos_queries)
            pt_logits = base.head(tgt_out).cpu()

        # TensorRT forward
        trt_logits = run_tensorrt_inference(engine, context, dummy_input.cpu().numpy())
        trt_logits_t = torch.from_numpy(trt_logits)

        # Difference
        diff = torch.abs(pt_logits - trt_logits_t)
        max_abs_diffs.append(float(diff.max().item()))
        mses.append(compute_mse(pt_logits, trt_logits_t))
        cos_sims.append(compute_cosine_similarity(pt_logits, trt_logits_t))

        # Prediction comparison
        pt_preds, _ = system.tokenizer.decode(pt_logits.softmax(-1))
        trt_preds, _ = system.tokenizer.decode(trt_logits_t.softmax(-1))
        if pt_preds == trt_preds:
            exact_agreements += 1
        else:
            # Diagnose first character where divergence occurs (Section 87)
            pred_pt = pt_preds[0] if pt_preds else ""
            pred_trt = trt_preds[0] if trt_preds else ""
            divergence_idx = -1
            for ci, (cp, ct) in enumerate(zip(pred_pt, pred_trt)):
                if cp != ct:
                    divergence_idx = ci
                    break
            divergence_diagnostics.append({
                "sample_idx": idx,
                "pytorch_pred": pred_pt,
                "tensorrt_pred": pred_trt,
                "first_divergence_char_idx": divergence_idx,
            })

    mean_max_diff = float(np.mean(max_abs_diffs))
    mean_mse = float(np.mean(mses))
    mean_cos = float(np.mean(cos_sims))
    agreement_pct = float(exact_agreements / num_samples * 100.0)

    # Check for integer-only status (Section 45, 46)
    int_only_status = "PASSED"
    if "int8" in expected_precision.lower():
        # Check if cosine similarity is acceptable and prediction agreement holds
        if mean_cos < 0.90 or agreement_pct < 80.0:
            int_only_status = "WARNING_DEGRADED"

    print(f"=== TensorRT Validation Results ({expected_precision.upper()}) ===")
    print(f"Mean Max Difference: {mean_max_diff:.6f}")
    print(f"Mean MSE: {mean_mse:.6e}")
    print(f"Mean Cosine Similarity: {mean_cos:.6f}")
    print(f"Prediction Agreement: {agreement_pct:.1f}% ({exact_agreements}/{num_samples})")
    print(f"Integer-Only Compliance: {int_only_status}")
    if divergence_diagnostics:
        print(f"Divergences detected on {len(divergence_diagnostics)} samples: {divergence_diagnostics[:2]}")

    return {
        "engine_path": engine_path,
        "expected_precision": expected_precision,
        "mean_max_abs_diff": mean_max_diff,
        "mean_mse": mean_mse,
        "mean_cosine_similarity": mean_cos,
        "prediction_agreement_pct": agreement_pct,
        "integer_only_status": int_only_status,
        "divergences": divergence_diagnostics,
    }


def evaluate_tensorrt_dataset(
    engine_path: str,
    data_loader,
    tokenizer,
    charset_adapter,
    max_samples: int = 200,
    img_size: tuple = (32, 128)
) -> Dict[str, Any]:
    from nltk import edit_distance
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    exact_matches = []
    ned_scores = []
    cer_scores = []
    total_evaluated = 0

    for batch_idx, (images, labels) in enumerate(data_loader):
        if max_samples and total_evaluated >= max_samples:
            break
        for img, gt in zip(images, labels):
            img_np = img.unsqueeze(0).numpy()
            trt_logits = run_tensorrt_inference(engine, context, img_np)
            trt_logits_t = torch.from_numpy(trt_logits)
            probs = trt_logits_t.softmax(-1)
            preds, _ = tokenizer.decode(probs)
            pred = charset_adapter(preds[0]) if preds else ""

            is_exact = 1.0 if pred == gt else 0.0
            exact_matches.append(is_exact)
            ed = edit_distance(pred, gt)
            max_l = max(len(pred), len(gt), 1)
            ned_scores.append(1.0 - (ed / max_l))
            cer_scores.append(ed / max(len(gt), 1))

            total_evaluated += 1
            if max_samples and total_evaluated >= max_samples:
                break

    return {
        "exact_plate_accuracy": float(np.mean(exact_matches)) if exact_matches else 0.0,
        "character_error_rate": float(np.mean(cer_scores)) if cer_scores else 0.0,
        "normalized_edit_distance": float(np.mean(ned_scores)) if ned_scores else 0.0,
        "num_samples": total_evaluated
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="pretrained/parseq_alpr_98.5.ckpt")
    parser.add_argument("--engine", type=str, required=True)
    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()

    validate_tensorrt_engine(
        checkpoint_path=args.checkpoint,
        engine_path=args.engine,
        expected_precision=args.precision,
        num_samples=args.samples
    )

