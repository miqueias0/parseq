# Scene Text Recognition Model Hub - Quantization Package
"""Specialized INT8 Quantization Suite for PARSeq.

Includes:
- Straight-Through Estimator (STE) for Quantization-Aware Fine-Tuning (QAT).
- Real Hardware INT8 Linear with dual-backend CUDA Tensor Cores and CPU oneDNN execution.
- Jetfire Per-Block Quantization and SmoothQuant Outlier Migration.
- Benchmarking and Accuracy Verification Engine.
"""

from .core import (
    QuantGranularity,
    SymmetricUniformQuantizer,
    PerBlockQuantizer,
    STEQuantizeFunction,
    ste_quantize,
)
from .layers import QuantizedLinear, RealHardwareInt8Linear, JetfireInt8Linear, JetfireFQTFunction
from .calibrator import ActivationCalibrator
from .smoothquant import compute_smooth_scale, apply_smoothquant_to_parseq
from .quantizer import PARSeqQuantizer
from .benchmark import BenchmarkEngine, BenchmarkMetrics

__all__ = [
    "QuantGranularity",
    "SymmetricUniformQuantizer",
    "PerBlockQuantizer",
    "STEQuantizeFunction",
    "ste_quantize",
    "QuantizedLinear",
    "RealHardwareInt8Linear",
    "JetfireInt8Linear",
    "JetfireFQTFunction",
    "ActivationCalibrator",
    "compute_smooth_scale",
    "apply_smoothquant_to_parseq",
    "PARSeqQuantizer",
    "BenchmarkEngine",
    "BenchmarkMetrics",
]
