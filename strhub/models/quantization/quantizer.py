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
    def prepare_for_int_flashattn(
        cls,
        model: nn.Module,
        block_size: int = 32,
        use_ibert_exp: bool = False,
        inplace: bool = False,
    ) -> nn.Module:
        """Equips PARSeq with INT-FlashAttention module (arXiv:2409.16997).
        
        Replaces decoder self-attention (with dynamic permutation & causal masks)
        and cross-attention (queries attending to ViT memory) with INT8MultiheadAttention.
        """
        from .int_attention import INT8MultiheadAttention

        m = model if inplace else copy.deepcopy(model)
        inner = cls._get_inner_model(m)

        if hasattr(inner, "decoder") and hasattr(inner.decoder, "layers"):
            for layer in inner.decoder.layers:
                if hasattr(layer, "self_attn") and isinstance(layer.self_attn, nn.MultiheadAttention):
                    layer.self_attn = INT8MultiheadAttention.from_float_mha(
                        layer.self_attn, block_size=block_size, use_ibert_exp=use_ibert_exp
                    )
                if hasattr(layer, "cross_attn") and isinstance(layer.cross_attn, nn.MultiheadAttention):
                    layer.cross_attn = INT8MultiheadAttention.from_float_mha(
                        layer.cross_attn, block_size=block_size, use_ibert_exp=use_ibert_exp
                    )

        log.info("Model successfully equipped with INT-FlashAttention.")
        return m

    @classmethod
    def prepare_for_ibert(
        cls,
        model: nn.Module,
        inplace: bool = False,
    ) -> nn.Module:
        """Configures PARSeq for pure integer-only arithmetic (I-BERT, ICML 2021).
        
        Replaces:
            - GELU with i-GELU (2nd-order polynomial approximation).
            - LayerNorm with ILayerNorm (Newton-Raphson integer square root).
            - Attention with I-BERT integer exponential bit-shifting.
        """
        from .ibert_ops import IGELU, ILayerNorm
        from .int_attention import INT8MultiheadAttention

        m = model if inplace else copy.deepcopy(model)
        inner = cls._get_inner_model(m)

        # 1. Encoder Blocks
        with torch.no_grad():
            if hasattr(inner, "encoder") and hasattr(inner.encoder, "blocks"):
                for block in inner.encoder.blocks:
                    if hasattr(block, "norm1") and isinstance(block.norm1, nn.LayerNorm):
                        iln = ILayerNorm(block.norm1.normalized_shape[0], eps=block.norm1.eps).to(block.norm1.weight.device)
                        iln.weight.copy_(block.norm1.weight)
                        iln.bias.copy_(block.norm1.bias)
                        block.norm1 = iln
                    if hasattr(block, "norm2") and isinstance(block.norm2, nn.LayerNorm):
                        iln = ILayerNorm(block.norm2.normalized_shape[0], eps=block.norm2.eps).to(block.norm2.weight.device)
                        iln.weight.copy_(block.norm2.weight)
                        iln.bias.copy_(block.norm2.bias)
                        block.norm2 = iln
                    if hasattr(block, "mlp") and hasattr(block.mlp, "act"):
                        block.mlp.act = IGELU()

                if hasattr(inner.encoder, "norm") and isinstance(inner.encoder.norm, nn.LayerNorm):
                    iln = ILayerNorm(inner.encoder.norm.normalized_shape[0], eps=inner.encoder.norm.eps).to(inner.encoder.norm.weight.device)
                    iln.weight.copy_(inner.encoder.norm.weight)
                    iln.bias.copy_(inner.encoder.norm.bias)
                    inner.encoder.norm = iln

            # 2. Decoder Layers
            if hasattr(inner, "decoder") and hasattr(inner.decoder, "layers"):
                for layer in inner.decoder.layers:
                    # Attention with i-exp bitshifts
                    if hasattr(layer, "self_attn") and isinstance(layer.self_attn, nn.MultiheadAttention):
                        layer.self_attn = INT8MultiheadAttention.from_float_mha(layer.self_attn, use_ibert_exp=True)
                    if hasattr(layer, "cross_attn") and isinstance(layer.cross_attn, nn.MultiheadAttention):
                        layer.cross_attn = INT8MultiheadAttention.from_float_mha(layer.cross_attn, use_ibert_exp=True)

                    # Norms
                    for norm_name in ["norm1", "norm2", "norm_q", "norm_c"]:
                        if hasattr(layer, norm_name) and isinstance(getattr(layer, norm_name), nn.LayerNorm):
                            ln = getattr(layer, norm_name)
                            iln = ILayerNorm(ln.normalized_shape[0], eps=ln.eps).to(ln.weight.device)
                            iln.weight.copy_(ln.weight)
                            iln.bias.copy_(ln.bias)
                            setattr(layer, norm_name, iln)

                    # GELU
                    layer.activation = IGELU()

                if hasattr(inner.decoder, "norm") and isinstance(inner.decoder.norm, nn.LayerNorm):
                    ln = inner.decoder.norm
                    iln = ILayerNorm(ln.normalized_shape[0], eps=ln.eps).to(ln.weight.device)
                    iln.weight.copy_(ln.weight)
                    iln.bias.copy_(ln.bias)
                    inner.decoder.norm = iln

        log.info("Model successfully configured for I-BERT Integer-Only execution.")
        return m

    @classmethod
    def prepare_for_unified_int8(
        cls,
        model: nn.Module,
        calibrator: Optional[ActivationCalibrator] = None,
        alpha: float = 0.5,
        block_size: int = 32,
        use_ibert_exp: bool = False,
        inplace: bool = False,
    ) -> nn.Module:
        """Unifies Jetfire INT8 Data Flow + INT-FlashAttention + Non-Linear Fused Operators.
        
        Step 1: Real Hardware INT8 Linear conversion for all GEMMs.
        Step 2: INT-FlashAttention for decoder self and cross attention.
        Step 3: Fused / Integer GELU and LayerNorm replacements.
        """
        m = model if inplace else copy.deepcopy(model)
        # 1. Convert Linear to Real INT8
        m = cls.convert_to_real_int8(m, calibrator=calibrator, alpha=alpha, apply_smooth=True, inplace=True)
        # 2. Convert Attention to INT-FlashAttention
        m = cls.prepare_for_int_flashattn(m, block_size=block_size, use_ibert_exp=use_ibert_exp, inplace=True)
        # 3. Convert Non-linears to Fused / Integer ops
        from .fused_ops import JetfireFusedGELU, JetfireFusedLayerNorm
        inner = cls._get_inner_model(m)

        with torch.no_grad():
            if hasattr(inner, "decoder") and hasattr(inner.decoder, "layers"):
                for layer in inner.decoder.layers:
                    layer.activation = JetfireFusedGELU(block_size=block_size)
                    for norm_name in ["norm1", "norm2", "norm_q", "norm_c"]:
                        if hasattr(layer, norm_name) and isinstance(getattr(layer, norm_name), nn.LayerNorm):
                            ln = getattr(layer, norm_name)
                            fln = JetfireFusedLayerNorm(ln.normalized_shape[0], eps=ln.eps, block_size=block_size).to(ln.weight.device)
                            fln.weight.copy_(ln.weight)
                            fln.bias.copy_(ln.bias)
                            setattr(layer, norm_name, fln)

        log.info("Model successfully converted to Unified INT8 Architecture (Jetfire + INT-FlashAttention + I-BERT).")
        return m

    @classmethod
    def quantize(
        cls,
        model: nn.Module,
        method: str = "real_int8",
        calibrator: Optional[ActivationCalibrator] = None,
        alpha: float = 0.5,
        use_per_block: bool = False,
        block_size: int = 32,
        use_ibert_exp: bool = False,
        inplace: bool = False,
    ) -> nn.Module:
        """Unified entry point for PARSeq quantization.
        
        Args:
            model: PARSeq model or LightningModule.
            method: Quantization strategy:
                - "real_int8": Native hardware INT8 with zero memory overhead.
                - "smoothquant_int8": SmoothQuant outlier migration + Real Hardware INT8.
                - "unified_int8": Jetfire INT8 Linear + INT-FlashAttention + Fused Non-linears.
                - "int_flashattn": INT-FlashAttention attention module.
                - "ibert": Integer-only i-GELU, i-Softmax, and i-LayerNorm.
                - "jetfire_fqt": Direct INT8 Fully Quantized Training (FQT) with INT8 forward & backward.
                - "qat": Quantization-Aware Training model for fine-tuning.
                - "dynamic": PyTorch standard dynamic INT8.
            calibrator: Calibration data statistics for SmoothQuant.
            alpha: SmoothQuant alpha hyperparameter.
            use_per_block: Whether to use Jetfire per-block tiling.
            block_size: Tile dimension for Jetfire per-block quantization.
            use_ibert_exp: Use I-BERT bit-shift exp in attention.
            inplace: Whether to modify model in place.
        """
        # Detect model device
        target_device = None
        for p in model.parameters():
            target_device = p.device
            break
        if target_device is None:
            for b in model.buffers():
                target_device = b.device
                break

        method = method.lower()
        if method == "real_int8":
            res = cls.convert_to_real_int8(model, calibrator=None, apply_smooth=False, inplace=inplace)
        elif method == "smoothquant_int8":
            res = cls.convert_to_real_int8(model, calibrator=calibrator, alpha=alpha, apply_smooth=True, inplace=inplace)
        elif method in ["unified_int8", "unified"]:
            res = cls.prepare_for_unified_int8(
                model, calibrator=calibrator, alpha=alpha, block_size=block_size, use_ibert_exp=use_ibert_exp, inplace=inplace
            )
        elif method in ["int_flashattn", "int_attention", "flashattn"]:
            res = cls.prepare_for_int_flashattn(model, block_size=block_size, use_ibert_exp=use_ibert_exp, inplace=inplace)
        elif method in ["ibert", "ibert_int8", "integer_only"]:
            res = cls.prepare_for_ibert(model, inplace=inplace)
        elif method == "qat":
            res = cls.prepare_for_qat(model, use_per_block=use_per_block, block_size=block_size, inplace=inplace)
        elif method in ["jetfire_fqt", "jetfire_training", "fqt"]:
            res = cls.prepare_for_jetfire_training(model, block_size=block_size, inplace=inplace)
        elif method == "dynamic":
            return cls.quantize_dynamic(model)
        else:
            valid = "['real_int8', 'smoothquant_int8', 'unified_int8', 'int_flashattn', 'ibert', 'jetfire_fqt', 'qat', 'dynamic']"
            raise ValueError(f"Unknown quantization method: {method}. Choose from {valid}")

        if target_device is not None and method != "dynamic":
            res = res.to(target_device)
        return res

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

    @classmethod
    def export_onnx_int8_qdq(
        cls,
        float_onnx_path: Union[str, Path],
        output_qdq_path: Union[str, Path],
        calibration_data_reader: Optional[object] = None,
        calib_dir: Optional[Union[str, Path]] = None,
        calib_samples: int = 64,
        calibrate_method: str = "MinMax",
        activation_type: str = "QInt8",
        per_channel: bool = True,
        op_types_to_quantize: Optional[List[str]] = None,
    ) -> Path:
        """Quantizes an ONNX model to Static QDQ INT8 format for NVIDIA TensorRT.
        
        Inserts QuantizeLinear and DequantizeLinear nodes which TensorRT fuses directly
        into high-throughput INT8 Tensor Core kernels on sm_75+ (Turing, Ampere, Ada, Hopper, Blackwell).
        Eliminates CPU-GPU Memcpy overhead completely.
        """
        import onnxruntime.quantization as ort_quant
        from .calibrator import PARSeqCalibrationDataReader

        float_onnx_path = Path(float_onnx_path)
        output_qdq_path = Path(output_qdq_path)
        output_qdq_path.parent.mkdir(parents=True, exist_ok=True)

        if calibration_data_reader is None:
            data_source = calib_dir if calib_dir is not None else "data/test"
            calibration_data_reader = PARSeqCalibrationDataReader(
                data_source=data_source,
                max_samples=calib_samples,
            )

        calib_method_enum = getattr(
            ort_quant.CalibrationMethod, calibrate_method, ort_quant.CalibrationMethod.MinMax
        )
        act_type_enum = getattr(
            ort_quant.QuantType, activation_type, ort_quant.QuantType.QInt8
        )

        if op_types_to_quantize is None:
            op_types_to_quantize = ["MatMul"]

        extra_opts = {
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
            "CalibTensorRangeSymmetric": True,
        }

        ort_quant.quantize_static(
            model_input=str(float_onnx_path),
            model_output=str(output_qdq_path),
            calibration_data_reader=calibration_data_reader,
            quant_format=ort_quant.QuantFormat.QDQ,
            activation_type=act_type_enum,
            weight_type=ort_quant.QuantType.QInt8,
            per_channel=per_channel,
            calibrate_method=calib_method_enum,
            op_types_to_quantize=op_types_to_quantize,
            extra_options=extra_opts,
        )

        # Run shape inference so all inserted QDQ nodes and boundary tensors have fully defined shapes for TensorRT
        import onnx
        from onnx import shape_inference

        try:
            qdq_model = onnx.load(str(output_qdq_path))
            inferred = shape_inference.infer_shapes(qdq_model, check_type=True)
            onnx.save(inferred, str(output_qdq_path))
            log.info(f"Shape inference completed for TensorRT on: {output_qdq_path}")
        except Exception as e:
            log.warning(f"Shape inference warning on QDQ model: {e}")

        log.info(f"Quantized Static QDQ INT8 ONNX model for TensorRT saved to: {output_qdq_path}")
        return output_qdq_path
