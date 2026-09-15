# Scene Text Recognition Model Hub - ONNX Export & Runtime Engine
# Copyright 2026 Darwin Bautista / PARSeq ONNX Extensions
#
# Licensed under the Apache License, Version 2.0 (the "License");

from .export import export_parseq_to_onnx, PARSeqExportWrapper
from .runtime import PARSeqONNXRuntime
from .benchmark import compare_pytorch_vs_onnx, format_onnx_comparison_table
from .evaluation import ONNXModelTestWrapper, evaluate_onnx_dataset, ONNXEvalResult
from .batch_benchmark import (
    BatchBenchmarkPoint,
    ModelBatchBenchmarkResult,
    benchmark_model_batch_scaling,
    run_batch_scaling_benchmark_suite,
    format_batch_scaling_markdown,
    discover_onnx_models,
    export_batch_benchmark_json,
)

__all__ = [
    "export_parseq_to_onnx",
    "PARSeqExportWrapper",
    "PARSeqONNXRuntime",
    "compare_pytorch_vs_onnx",
    "format_onnx_comparison_table",
    "ONNXModelTestWrapper",
    "evaluate_onnx_dataset",
    "ONNXEvalResult",
    "BatchBenchmarkPoint",
    "ModelBatchBenchmarkResult",
    "benchmark_model_batch_scaling",
    "run_batch_scaling_benchmark_suite",
    "format_batch_scaling_markdown",
    "discover_onnx_models",
    "export_batch_benchmark_json",
]
