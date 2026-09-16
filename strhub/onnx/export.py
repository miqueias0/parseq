# Scene Text Recognition Model Hub - ONNX Export & Runtime Engine
# Copyright 2026 Darwin Bautista / PARSeq ONNX Extensions
#
# Licensed under the Apache License, Version 2.0 (the "License");

import os
from typing import Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import onnx

from strhub.models.utils import load_from_checkpoint


import warnings


def disable_fused_attention(module: nn.Module) -> None:
    """Recursively disable fused SDPA in timm attention blocks.

    This ensures that scaled dot product attention uses static head_dim scaling
    rather than tracing dynamic tensor shape-slicing Cast nodes, eliminating
    the root cause of Cast output type mismatch in ONNX graphs.
    """
    for m in module.modules():
        if hasattr(m, "fused_attn"):
            m.fused_attn = False


def patch_fp16_cast_nodes(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
    """Repairs any Cast nodes where 'to' attribute was left as FLOAT (1) despite
    output tensor being converted to FLOAT16, and removes stale value_info.
    """
    graph = onnx_model.graph
    graph_output_names = {o.name for o in graph.output}

    for node in graph.node:
        if node.op_type == "Cast":
            # If the output is an internal tensor (not the final graph output), ensure it casts to FP16
            if node.output and node.output[0] not in graph_output_names:
                for attr in node.attribute:
                    if attr.name == "to" and attr.i == 1:  # 1 == FLOAT (FP32)
                        attr.i = 10  # 10 == FLOAT16

    # Clear stale value_info so ONNX Runtime can infer clean types dynamically
    del graph.value_info[:]
    return onnx_model


class PARSeqExportWrapper(nn.Module):
    """Wrapper to prepare PARSeq for clean, trace-friendly ONNX export."""

    def __init__(self, model: nn.Module, mode: str = "nar", max_length: Optional[int] = None, fp16: bool = False):
        super().__init__()
        self.model = model
        self.mode = mode
        self.fp16 = fp16
        base = getattr(model, "model", model)
        actual_max = getattr(base, "max_label_length", getattr(model, "max_label_length", getattr(getattr(model, "hparams", None), "max_label_length", 25)))
        self.max_length = max_length if max_length is not None else actual_max

        # Configure decode_ar on the underlying model
        if hasattr(self.model, "model") and hasattr(self.model.model, "decode_ar"):
            self.model.model.decode_ar = (mode == "ar")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.fp16:
            images = images.half()

        if self.mode == "nar":
            out = self.model(images)
        else:
            # Fixed length AR trace without dynamic python break
            out = self.model(images, max_length=self.max_length)

        if self.fp16:
            out = out.float()

        return out


def export_parseq_to_onnx(
    model_or_checkpoint: Union[nn.Module, str],
    output_path: str,
    mode: str = "nar",  # "nar" (recommended for low latency) or "ar"
    opset_version: int = 14,
    dynamic_batch: bool = True,
    fp16: bool = False,
    device: str = "cpu",
    sample_shape: Optional[Tuple[int, int, int, int]] = None,
    max_label_length: Optional[int] = None,
    charset_train: Optional[str] = None,
    **kwargs,
) -> str:
    """Exports a pretrained or fine-tuned PARSeq model to ONNX.

    Args:
        model_or_checkpoint: PARSeq PyTorch model or checkpoint path ('pretrained=parseq-tiny').
        output_path: Destination path for .onnx file.
        mode: 'nar' (Non-Autoregressive, fastest) or 'ar' (Autoregressive).
        opset_version: ONNX operator set version (default 14).
        dynamic_batch: Whether to allow variable batch size at runtime.
        fp16: Whether to convert the exported model to FP16 for CUDA acceleration.
        device: Device to run tracing on ('cpu' or 'cuda').
        sample_shape: Input tensor shape (batch, channels, height, width).
        max_label_length: Optional override for max output sequence length.
        charset_train: Optional override for train charset vocabulary.

    Returns:
        Absolute path to the exported ONNX model.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    extra_kwargs = dict(kwargs)
    if max_label_length is not None:
        extra_kwargs["max_label_length"] = max_label_length
    if charset_train is not None:
        extra_kwargs["charset_train"] = charset_train

    if isinstance(model_or_checkpoint, str):
        decode_ar = (mode == "ar")
        try:
            model = load_from_checkpoint(model_or_checkpoint, decode_ar=decode_ar, **extra_kwargs).eval()
        except Exception as e:
            if "size mismatch for pos_queries" in str(e) and "max_label_length" not in extra_kwargs:
                extra_kwargs["max_label_length"] = 25
                model = load_from_checkpoint(model_or_checkpoint, decode_ar=decode_ar, **extra_kwargs).eval()
            else:
                raise e
    else:
        model = model_or_checkpoint.eval()

    dev = torch.device(device)
    model = model.to(dev)

    # Disable timm fused attention so that attention uses clean static constant scaling
    disable_fused_attention(model)

    # Ensure decode_ar is properly configured on the model
    if hasattr(model, "model") and hasattr(model.model, "decode_ar"):
        model.model.decode_ar = (mode == "ar")

    # On CUDA, use native PyTorch FP16 tracing for superior numerical stability and clean graphs
    use_native_fp16 = fp16 and ("cuda" in device.lower())
    if use_native_fp16:
        model = model.half()

    wrapper = PARSeqExportWrapper(model, mode=mode, max_length=max_label_length, fp16=use_native_fp16)
    wrapper.eval().to(dev)

    img_size = model.hparams.img_size  # (H, W) e.g. (32, 128)
    if sample_shape is None:
        sample_shape = (1, 3, img_size[0], img_size[1])

    dummy_input = torch.randn(*sample_shape, device=dev, dtype=torch.float32)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "images": {0: "batch_size"},
            "logits": {0: "batch_size"},
        }

    # For AR mode, unrolled loop slices involve CPU index constants while masks are on CUDA.
    # PyTorch's _jit_pass_onnx_constant_fold has a known upstream bug causing:
    # "Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!".
    # Disabling PyTorch JIT constant folding in AR mode avoids this error; ONNX Runtime performs
    # full constant folding and graph optimizations when creating the InferenceSession.
    do_constant_folding = (mode != "ar")

    export_kwargs = dict(
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=do_constant_folding,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
    )

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            warnings.filterwarnings("ignore", category=UserWarning, module="torch.onnx")
            torch.onnx.export(
                wrapper,
                dummy_input,
                output_path,
                **export_kwargs,
            )
    except RuntimeError as e:
        if "constant_fold" in str(e) or "Expected all tensors to be on the same device" in str(e):
            warnings.warn(
                f"Constant folding failed ({e}). Retrying export with do_constant_folding=False. "
                "ONNX Runtime will optimize the graph at runtime."
            )
            export_kwargs["do_constant_folding"] = False
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
                warnings.filterwarnings("ignore", category=UserWarning, module="torch.onnx")
                torch.onnx.export(
                    wrapper,
                    dummy_input,
                    output_path,
                    **export_kwargs,
                )
        else:
            raise

    # Validate ONNX graph & infer complete shapes for TensorRT
    onnx_model = onnx.load(output_path)
    try:
        from onnx import shape_inference
        onnx_model = shape_inference.infer_shapes(onnx_model, check_type=True)
    except Exception:
        pass
    onnx.checker.check_model(onnx_model)

    # Add custom metadata properties
    base_m = getattr(model, "model", model)
    actual_max_len = getattr(base_m, "max_label_length", getattr(model, "max_label_length", getattr(getattr(model, "hparams", None), "max_label_length", 25)))
    actual_cs_train = getattr(model, "charset_train", getattr(getattr(model, "hparams", None), "charset_train", "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"))
    actual_cs_test = getattr(model, "charset_test", getattr(getattr(model, "hparams", None), "charset_test", "0123456789abcdefghijklmnopqrstuvwxyz"))

    meta_dict = {
        "model_architecture": "PARSeq",
        "decoding_mode": mode,
        "img_height": str(img_size[0]),
        "img_width": str(img_size[1]),
        "max_label_length": str(actual_max_len),
        "charset_train": str(actual_cs_train),
        "charset_test": str(actual_cs_test),
        "fp16": str(fp16),
    }
    for k, v in meta_dict.items():
        entry = onnx_model.metadata_props.add()
        entry.key = k
        entry.value = v

    onnx.save(onnx_model, output_path)

    # If FP16 was requested on CPU (where PyTorch LayerNorm doesn't support half),
    # use onnxconverter_common with our cast repair and disabled shape infer
    if fp16 and not use_native_fp16:
        try:
            from onnxconverter_common import float16
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                onnx_model_fp16 = float16.convert_float_to_float16(
                    onnx_model,
                    keep_io_types=True,
                    disable_shape_infer=True,
                )
            onnx_model_fp16 = patch_fp16_cast_nodes(onnx_model_fp16)
            onnx.save(onnx_model_fp16, output_path)
        except Exception as e:
            print(f"Warning: FP16 conversion failed ({e}). Keeping FP32 model.")

    return output_path
