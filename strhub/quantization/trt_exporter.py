"""
TensorRT Q/DQ ONNX Exporter and Engine Builder
==============================================
Exports PARSeq to ONNX with explicit QuantizeLinear and DequantizeLinear (Q/DQ) pairs
preserving NVIDIA TensorRT fusion patterns:
- QKV GEMM fusion
- SkipLayerNorm (Residual Add + LayerNorm) fusion
- Fast GELU fusion
And builds the TensorRT INT8 Engine (.engine / .plan) with dynamic shape profiles.
"""

import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from strhub.models.parseq.model import PARSeq


# ---------------------------------------------------------------------------
# Q/DQ Wrapper Modules for ONNX Export
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Q/DQ Wrapper Modules for ONNX Export
# ---------------------------------------------------------------------------

class QuantizeDequantizeFunc(torch.autograd.Function):
    """
    Autograd Function that exports native ONNX QuantizeLinear and DequantizeLinear nodes.
    Preserves exact NVIDIA TensorRT Q/DQ INT8 fusion patterns.
    """
    @staticmethod
    def forward(ctx, x: Tensor, scale: Tensor, axis: Optional[int] = None) -> Tensor:
        if axis == 0 and x.dim() >= 2:
            scale_b = scale.view(-1, *([1] * (x.dim() - 1)))
        else:
            scale_b = scale
        q = torch.clamp(torch.round(x / scale_b), -128, 127)
        return q * scale_b

    @staticmethod
    def symbolic(g, x, scale, axis: Optional[int] = None):
        zp = g.op("Constant", value_t=torch.tensor(0, dtype=torch.int8))
        if axis is not None:
            q = g.op("QuantizeLinear", x, scale, zp, axis_i=axis)
            dq = g.op("DequantizeLinear", q, scale, zp, axis_i=axis)
        else:
            q = g.op("QuantizeLinear", x, scale, zp)
            dq = g.op("DequantizeLinear", q, scale, zp)
        return dq


class QDQLinear(nn.Module):
    """
    Wraps nn.Linear with explicit QuantizeLinear -> DequantizeLinear for input and weights.
    Fuses into NVIDIA INT8 Tensor Core GEMM.
    """
    def __init__(self, linear: nn.Linear, scale_x: float = 0.05, scale_w: Optional[Tensor] = None):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = nn.Parameter(linear.weight.data.clone())
        self.bias = nn.Parameter(linear.bias.data.clone()) if linear.bias is not None else None

        sx = scale_x if isinstance(scale_x, torch.Tensor) else torch.tensor(scale_x, dtype=torch.float32)
        self.register_buffer("scale_x", sx.float().squeeze())

        if scale_w is None:
            max_w = self.weight.detach().abs().amax(dim=1)
            scale_w = torch.clamp(max_w / 127.0, min=1e-8)
        self.register_buffer("scale_w", scale_w.float().view(-1))

    def forward(self, x: Tensor) -> Tensor:
        x_dq = QuantizeDequantizeFunc.apply(x, self.scale_x)
        w_dq = QuantizeDequantizeFunc.apply(self.weight, self.scale_w, 0)
        return torch.nn.functional.linear(x_dq, w_dq, self.bias)


class QDQConv2d(nn.Module):
    """
    Wraps nn.Conv2d with explicit QuantizeLinear -> DequantizeLinear for input and weights.
    Fuses into NVIDIA INT8 Tensor Core Convolution.
    """
    def __init__(self, conv: nn.Conv2d, scale_x: float = 0.05, scale_w: Optional[Tensor] = None):
        super().__init__()
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.weight = nn.Parameter(conv.weight.data.clone())
        self.bias = nn.Parameter(conv.bias.data.clone()) if conv.bias is not None else None

        sx = scale_x if isinstance(scale_x, torch.Tensor) else torch.tensor(scale_x, dtype=torch.float32)
        self.register_buffer("scale_x", sx.float().squeeze())

        if scale_w is None:
            max_w = self.weight.detach().abs().amax(dim=(1, 2, 3))
            scale_w = torch.clamp(max_w / 127.0, min=1e-8)
        self.register_buffer("scale_w", scale_w.float().view(-1))

    def forward(self, x: Tensor) -> Tensor:
        x_dq = QuantizeDequantizeFunc.apply(x, self.scale_x)
        w_dq = QuantizeDequantizeFunc.apply(self.weight, self.scale_w, 0)
        return torch.nn.functional.conv2d(x_dq, w_dq, self.bias, self.stride, self.padding)


class QDQAttention(nn.Module):
    """
    Vision Transformer Attention block with explicit Q/DQ nodes around QKV GEMMs and BMMs.
    """
    def __init__(self, attn, scale_x: float = 0.05, scale_w_qkv: Optional[Tensor] = None, scale_w_proj: Optional[Tensor] = None):
        super().__init__()
        self.num_heads = attn.num_heads
        self.head_dim = attn.head_dim
        self.scale = attn.scale
        self.qkv = QDQLinear(attn.qkv, scale_x=scale_x, scale_w=scale_w_qkv)
        self.proj = QDQLinear(attn.proj, scale_x=scale_x, scale_w=scale_w_proj)
        s_act = scale_x if isinstance(scale_x, torch.Tensor) else torch.tensor(scale_x, dtype=torch.float32)
        self.register_buffer("scale_act", s_act.clone().detach().float().squeeze())

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = q * self.scale
        q_dq = QuantizeDequantizeFunc.apply(q, self.scale_act)
        k_dq = QuantizeDequantizeFunc.apply(k, self.scale_act)
        attn = (q_dq @ k_dq.transpose(-2, -1)).softmax(dim=-1)
        attn_dq = QuantizeDequantizeFunc.apply(attn, self.scale_act)
        v_dq = QuantizeDequantizeFunc.apply(v, self.scale_act)
        x = attn_dq @ v_dq
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class QDQMlp(nn.Module):
    """
    Vision Transformer MLP block with explicit Q/DQ nodes.
    """
    def __init__(self, mlp, scale_x: float = 0.05, scale_w_fc1: Optional[Tensor] = None, scale_w_fc2: Optional[Tensor] = None):
        super().__init__()
        self.fc1 = QDQLinear(mlp.fc1, scale_x=scale_x, scale_w=scale_w_fc1)
        self.act = mlp.act
        self.fc2 = QDQLinear(mlp.fc2, scale_x=scale_x, scale_w=scale_w_fc2)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class QDQBlock(nn.Module):
    """
    Vision Transformer Block with Q/DQ layers.
    """
    def __init__(self, blk, scale_x: float = 0.05, scales: Optional[dict] = None):
        super().__init__()
        scales = scales or {}
        self.norm1 = blk.norm1
        self.attn = QDQAttention(
            blk.attn,
            scale_x=scales.get("attn.qkv.scale_x", scale_x),
            scale_w_qkv=scales.get("attn.qkv.scale_w", None),
            scale_w_proj=scales.get("attn.proj.scale_w", None),
        )
        self.norm2 = blk.norm2
        self.mlp = QDQMlp(
            blk.mlp,
            scale_x=scales.get("mlp.fc1.scale_x", scale_x),
            scale_w_fc1=scales.get("mlp.fc1.scale_w", None),
            scale_w_fc2=scales.get("mlp.fc2.scale_w", None),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class QDQViTEncoder(nn.Module):
    """
    Vision Transformer Encoder containing 100% native Q/DQ nodes for TensorRT INT8 optimization.
    """
    def __init__(self, base_encoder, state_dict: Optional[dict] = None, scale_dict: Optional[dict] = None):
        super().__init__()
        state_dict = state_dict or {}
        scale_dict = scale_dict or {}

        proj_sx = state_dict.get("encoder.patch_embed.proj.scale_x", 0.05)
        proj_sw = state_dict.get("encoder.patch_embed.proj.scale_w", None)
        self.patch_embed_proj = QDQConv2d(base_encoder.patch_embed.proj, scale_x=proj_sx, scale_w=proj_sw)
        self.pos_embed = nn.Parameter(base_encoder.pos_embed.data.clone())

        blocks = []
        for i, blk in enumerate(base_encoder.blocks):
            blk_scales = {
                "attn.qkv.scale_x": state_dict.get(f"encoder.blocks.{i}.attn.qkv.scale_x", 0.05),
                "attn.qkv.scale_w": state_dict.get(f"encoder.blocks.{i}.attn.qkv.scale_w", None),
                "attn.proj.scale_w": state_dict.get(f"encoder.blocks.{i}.attn.proj.scale_w", None),
                "mlp.fc1.scale_x": state_dict.get(f"encoder.blocks.{i}.mlp.fc1.scale_x", 0.05),
                "mlp.fc1.scale_w": state_dict.get(f"encoder.blocks.{i}.mlp.fc1.scale_w", None),
                "mlp.fc2.scale_w": state_dict.get(f"encoder.blocks.{i}.mlp.fc2.scale_w", None),
            }
            blocks.append(QDQBlock(blk, scales=blk_scales))
        self.blocks = nn.ModuleList(blocks)
        self.norm = base_encoder.norm

    def forward(self, img: Tensor) -> Tensor:
        x = self.patch_embed_proj(img)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


# ---------------------------------------------------------------------------
# TensorRT ONNX Exporter
# ---------------------------------------------------------------------------

class TensorRTExporter:
    """
    Converts PARSeq to ONNX with explicit Q/DQ pairing and compiles to TensorRT INT8 engine.
    """
    def __init__(self, model: nn.Module, checkpoint_state: Optional[dict] = None, scale_dict: Optional[dict] = None):
        if hasattr(model, "model") and not hasattr(model, "encode"):
            self.model = model.model
        else:
            self.model = model
        self.checkpoint_state = checkpoint_state or {}
        self.scale_dict = scale_dict or {}

    def export_onnx(
        self,
        output_path: str,
        input_shape: Tuple[int, int, int, int] = (1, 3, 32, 128),
        opset_version: int = 17,
        device: str = "cpu",
    ) -> str:
        """
        Exports the model encoder to ONNX format with explicit QuantizeLinear/DequantizeLinear nodes.
        """
        self.model.eval()
        self.model.to(device)

        dummy_img = torch.randn(*input_shape, device=device)
        print(f"[*] Exporting PARSeq explicit QDQ Encoder to ONNX at {output_path} (opset {opset_version})...")

        qdq_encoder = QDQViTEncoder(self.model.encoder, state_dict=self.checkpoint_state, scale_dict=self.scale_dict)
        qdq_encoder.eval().to(device)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        torch.onnx.export(
            qdq_encoder,
            dummy_img,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["images"],
            output_names=["memory"],
            dynamic_axes={
                "images": {0: "batch_size"},
                "memory": {0: "batch_size"},
            },
        )
        print(f"[+] Successfully exported ONNX QDQ graph to: {output_path}")
        return output_path

    def build_trt_engine(
        self,
        onnx_path: str,
        engine_path: str,
        min_shape: Tuple[int, int, int, int] = (1, 3, 32, 128),
        opt_shape: Tuple[int, int, int, int] = (8, 3, 32, 128),
        max_shape: Tuple[int, int, int, int] = (32, 3, 32, 128),
        int8_mode: bool = True,
    ) -> Optional[str]:
        """
        Compiles the ONNX model into a high-performance TensorRT INT8 Engine.
        """
        try:
            import tensorrt as trt
        except ImportError:
            print("[!] tensorrt module not installed or unavailable. Skipping engine compilation.")
            return None

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(TRT_LOGGER)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        config = builder.create_builder_config()
        parser = trt.OnnxParser(network, TRT_LOGGER)

        print(f"[*] Reading ONNX model from {onnx_path}...")
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    print("[!] TensorRT Parser Error:", parser.get_error(error))
                return None

        has_qdq = getattr(network, "has_explicit_quantization", False)
        print(f"[*] Network explicit Q/DQ quantization detected: {has_qdq}")

        # Enable INT8 and FP16 builder flags
        if int8_mode and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print("[+] TensorRT INT8 mode enabled.")

            # If network does not have explicit Q/DQ, provide dynamic ranges to prevent builder error
            if not has_qdq:
                print("[*] Applying fallback tensor dynamic ranges for implicit INT8 calibration...")
                for i in range(network.num_inputs):
                    inp = network.get_input(i)
                    if inp.is_execution_tensor and not inp.dynamic_range:
                        inp.dynamic_range = (-1.0, 1.0)
                for i in range(network.num_layers):
                    layer = network.get_layer(i)
                    for j in range(layer.num_outputs):
                        out_t = layer.get_output(j)
                        if out_t.is_execution_tensor and not out_t.dynamic_range:
                            out_t.dynamic_range = (-16.0, 16.0)

        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[+] TensorRT FP16 fallback mode enabled.")

        # Configure dynamic shape profile
        profile = builder.create_optimization_profile()
        profile.set_shape("images", min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)

        # Set workspace memory limit (2GB)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * (1024 ** 3))

        print(f"[*] Building serialized TensorRT engine at {engine_path}...")
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            print("[!] Failed to build TensorRT engine.")
            return None

        os.makedirs(os.path.dirname(os.path.abspath(engine_path)), exist_ok=True)
        with open(engine_path, "wb") as f:
            f.write(serialized_engine)

        print(f"[+] TensorRT engine successfully saved to: {engine_path}")
        return engine_path


def export_onnx_qdq(
    model: PARSeq,
    output_path: str,
    input_shape: Tuple[int, int, int, int] = (1, 3, 32, 128),
    opset_version: int = 17,
    device: str = "cpu",
) -> str:
    exporter = TensorRTExporter(model)
    return exporter.export_onnx(output_path, input_shape=input_shape, opset_version=opset_version, device=device)


def build_tensorrt_engine(
    onnx_path: str,
    engine_path: str,
    int8_mode: bool = True,
) -> Optional[str]:
    # Dummy PARSeq model to use class helper
    try:
        import tensorrt as trt
    except ImportError:
        print("[!] TensorRT is not available.")
        return None

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print("[!] TRT Parser Error:", parser.get_error(error))
            return None

    if int8_mode and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    profile.set_shape("images", (1, 3, 32, 128), (8, 3, 32, 128), (32, 3, 32, 128))
    config.add_optimization_profile(profile)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * (1024 ** 3))

    serialized = builder.build_serialized_network(network, config)
    if serialized:
        os.makedirs(os.path.dirname(os.path.abspath(engine_path)), exist_ok=True)
        with open(engine_path, "wb") as f:
            f.write(serialized)
        return engine_path
    return None
