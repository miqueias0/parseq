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
from .layers import QuantizedLinear, RealHardwareInt8Linear, JetfireInt8Linear, JetfireFQTFunction, block_quantize_2d, dequantize_blocks
from .ibert_ops import IGELU, IExpSoftmax, ILayerNorm, integer_sqrt_newton_raphson
from .int_attention import INT8MultiheadAttention, int_flash_attention_core, quantize_token_symmetric
from .fused_ops import JetfireFusedGELU, JetfireFusedLayerNorm, JetfireFusedResidualAdd
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
    "block_quantize_2d",
    "dequantize_blocks",
    "IGELU",
    "IExpSoftmax",
    "ILayerNorm",
    "integer_sqrt_newton_raphson",
    "INT8MultiheadAttention",
    "int_flash_attention_core",
    "quantize_token_symmetric",
    "JetfireFusedGELU",
    "JetfireFusedLayerNorm",
    "JetfireFusedResidualAdd",
    "ActivationCalibrator",
    "compute_smooth_scale",
    "apply_smoothquant_to_parseq",
    "PARSeqQuantizer",
    "BenchmarkEngine",
    "BenchmarkMetrics",
]
