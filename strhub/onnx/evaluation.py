# Scene Text Recognition Model Hub - ONNX Export & Runtime Engine
# Copyright 2026 Darwin Bautista / PARSeq ONNX Extensions
#
# Licensed under the Apache License, Version 2.0 (the "License");

from argparse import Namespace
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import string
import time
import numpy as np
from tqdm import tqdm
from nltk import edit_distance
import torch

from strhub.data.utils import CharsetAdapter
from strhub.models.base import BatchResult
from .runtime import PARSeqONNXRuntime


@dataclass
class ONNXEvalResult:
    dataset: str
    num_samples: int
    accuracy: float
    ned: float
    confidence: float
    label_length: float
    total_time_s: float
    mean_batch_ms: float
    fps: float


class ONNXModelTestWrapper:
    """Drop-in adapter wrapping PARSeqONNXRuntime to mimic a PyTorch Lightning model's
    evaluation interface. Allows test.py to run directly on .onnx models without modifications.
    """

    def __init__(
        self,
        onnx_model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        device_id: int = 0,
        charset_test: Optional[str] = None,
    ):
        self.engine = PARSeqONNXRuntime(onnx_model_path, device=device, device_id=device_id)
        if charset_test is not None:
            self.engine.charset_test = charset_test
            self.engine.charset_adapter = CharsetAdapter(charset_test)

        self.hparams = Namespace(
            img_size=self.engine.img_size,
            max_label_length=self.engine.max_label_length,
            charset_train=self.engine.charset_train,
            charset_test=self.engine.charset_test,
        )
        self.device = torch.device(device if torch.cuda.is_available() and "cuda" in device else "cpu")

    def test_step(self, batch: Tuple[torch.Tensor, Sequence[str]], batch_idx: int = -1) -> Dict[str, BatchResult]:
        """Evaluates a batch of (images, ground_truth_labels) and returns ICDAR metrics."""
        images, labels = batch
        logits = self.engine.forward(images)
        logits_tensor = torch.from_numpy(logits).float()
        probs = logits_tensor.softmax(-1)
        preds, probs = self.engine.tokenizer.decode(probs)

        total = 0
        correct = 0
        ned = 0.0
        confidence = 0.0
        label_length = 0

        for pred, prob, gt in zip(preds, probs, labels):
            confidence += prob.prod().item() if len(prob) > 0 else 0.0
            pred = self.engine.charset_adapter(pred)
            denom = max(len(pred), len(gt))
            ned += (edit_distance(pred, gt) / denom) if denom > 0 else 0.0
            if pred == gt:
                correct += 1
            total += 1
            label_length += len(pred)

        return {"output": BatchResult(total, correct, ned, confidence, label_length, None, None)}

    def eval(self):
        """No-op for PyTorch interface compatibility."""
        return self

    def to(self, device):
        """No-op for PyTorch interface compatibility."""
        return self


def evaluate_onnx_dataset(
    engine: PARSeqONNXRuntime,
    dataloader,
    dataset_name: str = "Dataset",
) -> ONNXEvalResult:
    """Evaluates an ONNX Runtime engine over a DataLoader and computes accuracy, NED, and FPS."""
    total = 0
    correct = 0
    ned = 0.0
    confidence = 0.0
    label_length = 0

    batch_times = []
    t_start = time.perf_counter()

    for imgs, labels in tqdm(dataloader, desc=f"Evaluating {dataset_name}", leave=False):
        b_start = time.perf_counter_ns()
        logits = engine.forward(imgs)
        batch_times.append((time.perf_counter_ns() - b_start) / 1e6)

        logits_tensor = torch.from_numpy(logits).float()
        probs = logits_tensor.softmax(-1)
        preds, probs = engine.tokenizer.decode(probs)

        for pred, prob, gt in zip(preds, probs, labels):
            confidence += prob.prod().item() if len(prob) > 0 else 0.0
            pred = engine.charset_adapter(pred)
            denom = max(len(pred), len(gt))
            ned += (edit_distance(pred, gt) / denom) if denom > 0 else 0.0
            if pred == gt:
                correct += 1
            total += 1
            label_length += len(pred)

    total_time_s = time.perf_counter() - t_start
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    mean_ned = 100.0 * (1.0 - ned / total) if total > 0 else 0.0
    mean_conf = 100.0 * confidence / total if total > 0 else 0.0
    mean_len = label_length / total if total > 0 else 0.0
    mean_batch_ms = float(np.mean(batch_times)) if batch_times else 0.0
    fps = total / total_time_s if total_time_s > 0 else 0.0

    return ONNXEvalResult(
        dataset=dataset_name,
        num_samples=total,
        accuracy=accuracy,
        ned=mean_ned,
        confidence=mean_conf,
        label_length=mean_len,
        total_time_s=total_time_s,
        mean_batch_ms=mean_batch_ms,
        fps=fps,
    )
