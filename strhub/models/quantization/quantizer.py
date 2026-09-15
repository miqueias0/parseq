# Scene Text Recognition Model Hub - PARSeq INT8 Quantizer
# Core Orchestration Engine for PARSeq Quantization, Fine-Tuning (QAT), and Export

import copy
import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any, List

import torch
import torch.nn as nn

from .calibrator import ActivationCalibrator
from .core import QuantGranularity
from .layers import QuantizedLinear, RealHardwareInt8Linear, JetfireInt8Linear
from .smoothquant import apply_smoothquant_to_parseq

log = logging.getLogger("PARSeqQuantizer")


class PARSeqQuantizer:
    """Specialized Quantizer Engine for PARSeq (Scene Text Recognition).
    
    Provides end-to-end support for:
        1. Quantization-Aware Training (QAT) preparation for fine-tuning.
        2. Real Hardware INT8 conversion (CUDA Tensor Cores & CPU oneDNN).
        3. SmoothQuant outlier migration.
        4. Standard PyTorch dynamic quantization.
        5. Optimized ONNX INT8 export.
    """

    @staticmethod
    def _get_inner_model(model: nn.Module) -> nn.Module:
        """Extracts the underlying PARSeq model from LightningModule or wrapper."""
        return getattr(model, "model", model)

    @classmethod
    def prepare_for_qat(
        cls,
        model: nn.Module,
        use_per_block: bool = False,
        block_size: int = 64,
        inplace: bool = False,
    ) -> nn.Module:
        """Converts PARSeq Linear layers to QuantizedLinear for QAT fine-tuning.
        
        Preserves trainable weight parameters while enabling Straight-Through Estimator (STE)
        so that standard loss backpropagation and optimizer steps update weights with INT8 constraints.
        """
        m = model if inplace else copy.deepcopy(model)
        inner = cls._get_inner_model(m)

        # 1. Encoder ViT Blocks
        if hasattr(inner, "encoder") and hasattr(inner.encoder, "blocks"):
            for block in inner.encoder.blocks:
                if hasattr(block, "attn"):
                    if hasattr(block.attn, "qkv") and isinstance(block.attn.qkv, nn.Linear):
                        block.attn.qkv = QuantizedLinear.from_float(
                            block.attn.qkv, use_per_block=use_per_block, block_size=block_size
                        )
                    if hasattr(block.attn, "proj") and isinstance(block.attn.proj, nn.Linear):
                        block.attn.proj = QuantizedLinear.from_float(
                            block.attn.proj, use_per_block=use_per_block, block_size=block_size
                        )
                if hasattr(block, "mlp"):
                    if hasattr(block.mlp, "fc1") and isinstance(block.mlp.fc1, nn.Linear):
                        block.mlp.fc1 = QuantizedLinear.from_float(
                            block.mlp.fc1, use_per_block=use_per_block, block_size=block_size
                        )
                    if hasattr(block.mlp, "fc2") and isinstance(block.mlp.fc2, nn.Linear):
                        block.mlp.fc2 = QuantizedLinear.from_float(
                            block.mlp.fc2, use_per_block=use_per_block, block_size=block_size
                        )

        # 2. Decoder Layers
        if hasattr(inner, "decoder") and hasattr(inner.decoder, "layers"):
            for layer in inner.decoder.layers:
                if hasattr(layer, "linear1") and isinstance(layer.linear1, nn.Linear):
                    layer.linear1 = QuantizedLinear.from_float(
                        layer.linear1, use_per_block=use_per_block, block_size=block_size
                    )
                if hasattr(layer, "linear2") and isinstance(layer.linear2, nn.Linear):
                    layer.linear2 = QuantizedLinear.from_float(
                        layer.linear2, use_per_block=use_per_block, block_size=block_size
                    )

        # 3. Head Linear
        if hasattr(inner, "head") and isinstance(inner.head, nn.Linear):
            inner.head = QuantizedLinear.from_float(
                inner.head, use_per_block=use_per_block, block_size=block_size
            )

        log.info("Model successfully prepared for Quantization-Aware Fine-Tuning (QAT).")
        return m

    @classmethod
    def convert_to_real_int8(
        cls,
        model: nn.Module,
        calibrator: Optional[ActivationCalibrator] = None,
        alpha: float = 0.5,
        apply_smooth: bool = True,
        inplace: bool = False,
    ) -> nn.Module:
        """Converts PARSeq to Real Hardware INT8 Execution (CUDA & CPU).
        
        Step 1: Optionally apply SmoothQuant parameter migration to resolve activation spikes.
        Step 2: Replace Linear layers with RealHardwareInt8Linear (weights stored as torch.int8).
        """
        m = model if inplace else copy.deepcopy(model)
        if apply_smooth:
            m = apply_smoothquant_to_parseq(m, calibrator=calibrator, alpha=alpha, inplace=True)

        inner = cls._get_inner_model(m)

        # 1. Encoder ViT Blocks
        if hasattr(inner, "encoder") and hasattr(inner.encoder, "blocks"):
            for block in inner.encoder.blocks:
                if hasattr(block, "attn"):
                    if hasattr(block.attn, "qkv") and isinstance(block.attn.qkv, (nn.Linear, QuantizedLinear)):
                        block.attn.qkv = RealHardwareInt8Linear.from_float(block.attn.qkv)
                    if hasattr(block.attn, "proj") and isinstance(block.attn.proj, (nn.Linear, QuantizedLinear)):
                        block.attn.proj = RealHardwareInt8Linear.from_float(block.attn.proj)
                if hasattr(block, "mlp"):
                    if hasattr(block.mlp, "fc1") and isinstance(block.mlp.fc1, (nn.Linear, QuantizedLinear)):
                        block.mlp.fc1 = RealHardwareInt8Linear.from_float(block.mlp.fc1)
                    if hasattr(block.mlp, "fc2") and isinstance(block.mlp.fc2, (nn.Linear, QuantizedLinear)):
                        block.mlp.fc2 = RealHardwareInt8Linear.from_float(block.mlp.fc2)

        # 2. Decoder Layers
        if hasattr(inner, "decoder") and hasattr(inner.decoder, "layers"):
            for layer in inner.decoder.layers:
                if hasattr(layer, "linear1") and isinstance(layer.linear1, (nn.Linear, QuantizedLinear)):
                    layer.linear1 = RealHardwareInt8Linear.from_float(layer.linear1)
                if hasattr(layer, "linear2") and isinstance(layer.linear2, (nn.Linear, QuantizedLinear)):
                    layer.linear2 = RealHardwareInt8Linear.from_float(layer.linear2)

        # 3. Head Linear
        if hasattr(inner, "head") and isinstance(inner.head, (nn.Linear, QuantizedLinear)):
            inner.head = RealHardwareInt8Linear.from_float(inner.head)

        log.info("Model successfully converted to Real Hardware INT8 (CUDA & CPU).")
        return m

    @classmethod
    def quantize_dynamic(cls, model: nn.Module) -> nn.Module:
        """Applies PyTorch standard dynamic quantization for CPU."""
        m = copy.deepcopy(model).cpu()
        quantized = torch.ao.quantization.quantize_dynamic(
            m, {nn.Linear}, dtype=torch.qint8
        )
        return quantized

    @classmethod
    def prepare_for_jetfire_training(
        cls,
        model: nn.Module,
        block_size: int = 64,
        inplace: bool = False,
    ) -> nn.Module:
        """Converts PARSeq Linear layers to JetfireInt8Linear for direct INT8 training (FQT).
        
        Enables 8-bit data flow for both forward activations and backward gradients,
        with per-block quantization (Xi et al., Jetfire ICML 2024).
        """
        m = model if inplace else copy.deepcopy(model)
        inner = cls._get_inner_model(m)

        # 1. Encoder ViT Blocks
        if hasattr(inner, "encoder") and hasattr(inner.encoder, "blocks"):
            for block in inner.encoder.blocks:
                if hasattr(block, "attn"):
                    if hasattr(block.attn, "qkv") and isinstance(block.attn.qkv, (nn.Linear, QuantizedLinear)):
                        block.attn.qkv = JetfireInt8Linear.from_float(block.attn.qkv, block_size=block_size)
                    if hasattr(block.attn, "proj") and isinstance(block.attn.proj, (nn.Linear, QuantizedLinear)):
                        block.attn.proj = JetfireInt8Linear.from_float(block.attn.proj, block_size=block_size)
                if hasattr(block, "mlp"):
                    if hasattr(block.mlp, "fc1") and isinstance(block.mlp.fc1, (nn.Linear, QuantizedLinear)):
                        block.mlp.fc1 = JetfireInt8Linear.from_float(block.mlp.fc1, block_size=block_size)
                    if hasattr(block.mlp, "fc2") and isinstance(block.mlp.fc2, (nn.Linear, QuantizedLinear)):
                        block.mlp.fc2 = JetfireInt8Linear.from_float(block.mlp.fc2, block_size=block_size)

        # 2. Decoder Layers
        if hasattr(inner, "decoder") and hasattr(inner.decoder, "layers"):
            for layer in inner.decoder.layers:
                if hasattr(layer, "linear1") and isinstance(layer.linear1, (nn.Linear, QuantizedLinear)):
                    layer.linear1 = JetfireInt8Linear.from_float(layer.linear1, block_size=block_size)
                if hasattr(layer, "linear2") and isinstance(layer.linear2, (nn.Linear, QuantizedLinear)):
                    layer.linear2 = JetfireInt8Linear.from_float(layer.linear2, block_size=block_size)

        # 3. Head Linear
        if hasattr(inner, "head") and isinstance(inner.head, (nn.Linear, QuantizedLinear)):
            inner.head = JetfireInt8Linear.from_float(inner.head, block_size=block_size)

        log.info("Model successfully prepared for Jetfire Direct INT8 Training (FQT).")
        return m

    @classmethod
    def quantize(
        cls,
        model: nn.Module,
        method: str = "real_int8",
        calibrator: Optional[ActivationCalibrator] = None,
        alpha: float = 0.5,
        use_per_block: bool = False,
        block_size: int = 64,
        inplace: bool = False,
    ) -> nn.Module:
        """Unified entry point for PARSeq quantization.
        
        Args:
            model: PARSeq model or LightningModule.
            method: Quantization strategy:
                - "real_int8": Native hardware INT8 with zero memory overhead.
                - "smoothquant_int8": SmoothQuant outlier migration + Real Hardware INT8.
                - "qat": Quantization-Aware Training model for fine-tuning.
                - "jetfire_fqt": Direct INT8 Fully Quantized Training (FQT) with INT8 forward & backward.
                - "dynamic": PyTorch standard dynamic INT8.
            calibrator: Calibration data statistics for SmoothQuant.
            alpha: SmoothQuant alpha hyperparameter.
            use_per_block: Whether to use Jetfire per-block tiling.
            block_size: Tile dimension for Jetfire per-block quantization.
            inplace: Whether to modify model in place.
        """
        method = method.lower()
        if method == "real_int8":
            return cls.convert_to_real_int8(model, calibrator=None, apply_smooth=False, inplace=inplace)
        elif method == "smoothquant_int8":
            return cls.convert_to_real_int8(model, calibrator=calibrator, alpha=alpha, apply_smooth=True, inplace=inplace)
        elif method == "qat":
            return cls.prepare_for_qat(model, use_per_block=use_per_block, block_size=block_size, inplace=inplace)
        elif method in ["jetfire_fqt", "jetfire_training", "fqt"]:
            return cls.prepare_for_jetfire_training(model, block_size=block_size, inplace=inplace)
        elif method == "dynamic":
            return cls.quantize_dynamic(model)
        else:
            raise ValueError(f"Unknown quantization method: {method}. Choose from ['real_int8', 'smoothquant_int8', 'qat', 'jetfire_fqt', 'dynamic']")

    @classmethod
    def export_onnx(
        cls,
        model_or_checkpoint: Union[nn.Module, str],
        output_path: Union[str, Path],
        img_size: tuple = (32, 128),
        mode: str = "nar",
        device: str = "cpu",
        opset_version: int = 14,
        **kwargs,
    ) -> Path:
        """Exports PARSeq model to ONNX with proper input signatures and low-latency NAR decoding."""
        from strhub.onnx.export import export_parseq_to_onnx

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        res = export_parseq_to_onnx(
            model_or_checkpoint=model_or_checkpoint,
            output_path=str(output_path),
            mode=mode,
            opset_version=opset_version,
            device=device,
            **kwargs,
        )
        log.info(f"Exported ONNX model ({mode.upper()} mode) to: {output_path}")
        return Path(res)

    @classmethod
    def export_onnx_int8(
        cls,
        float_onnx_path: Union[str, Path],
        output_int8_path: Union[str, Path],
        per_channel: bool = True,
        op_types_to_quantize: Optional[List[str]] = None,
    ) -> Path:
        """Quantizes an ONNX model to INT8 using ONNX Runtime.
        
        Uses real QLinearMatMul / MatMulInteger execution nodes for maximum physical speedup.
        Defaults to op_types_to_quantize=['MatMul'] to ensure universal compatibility on both
        CPU and CUDA without missing kernel errors.
        """
        import onnxruntime.quantization as ort_quant

        float_onnx_path = Path(float_onnx_path)
        output_int8_path = Path(output_int8_path)
        output_int8_path.parent.mkdir(parents=True, exist_ok=True)

        if op_types_to_quantize is None:
            op_types_to_quantize = ["MatMul"]

        ort_quant.quantize_dynamic(
            model_input=str(float_onnx_path),
            model_output=str(output_int8_path),
            op_types_to_quantize=op_types_to_quantize,
            per_channel=per_channel,
            weight_type=ort_quant.QuantType.QInt8,
        )
        log.info(f"Quantized ONNX INT8 model saved to: {output_int8_path}")
        return output_int8_path
