"""
PARSeq Integer-Only & Accelerated INT8 Quantization Framework
=============================================================
Integrates:
- IPTQ-ViT: Data-aware Poly-GELU, Efficient Bit-Softmax, and Unified Metric (Omega)
- I-BERT: Integer-only arithmetic, Integer LayerNorm with Newton integer sqrt
- HAWQ-V3: Dyadic static rescaling and Dyadic Residual Addition
- Quant-Noise: Stochastic fake-quantization fine-tuning
- Q8BERT / Transformer-LT: Symmetric per-channel weight, per-tensor KL activation calibration
- TensorRT: Q/DQ ONNX export with QKV GEMM and SkipLayerNorm fusion patterns
"""

from .core import (
    MinMaxObserver,
    PerChannelMinMaxObserver,
    KLHistogramObserver,
    QuantNoiseLinear,
    QuantNoiseConv2d,
    QuantizedPatchEmbed,
    DataAwarePolyGELU,
    EfficientBitSoftmax,
    IntegerLayerNorm,
    DyadicResidualAdd,
    IntegerLinear,
    IntegerMatMul,
    float_to_dyadic,
    dyadic_scale,
    integer_sqrt_newton,
)

from .unified_metric import (
    UnifiedMetricCalculator,
    UnifiedMetricSearcher,
    evaluate_operator_metrics,
)

from .parseq_quantizer import (
    PARSeqQuantizer,
    quantize_parseq,
)

from .trt_exporter import (
    TensorRTExporter,
    export_onnx_qdq,
    build_tensorrt_engine,
)

__all__ = [
    "MinMaxObserver",
    "PerChannelMinMaxObserver",
    "KLHistogramObserver",
    "QuantNoiseLinear",
    "QuantNoiseConv2d",
    "QuantizedPatchEmbed",
    "DataAwarePolyGELU",
    "EfficientBitSoftmax",
    "IntegerLayerNorm",
    "DyadicResidualAdd",
    "IntegerLinear",
    "IntegerMatMul",
    "float_to_dyadic",
    "dyadic_scale",
    "integer_sqrt_newton",
    "UnifiedMetricCalculator",
    "UnifiedMetricSearcher",
    "evaluate_operator_metrics",
    "PARSeqQuantizer",
    "quantize_parseq",
    "TensorRTExporter",
    "export_onnx_qdq",
    "build_tensorrt_engine",
]
