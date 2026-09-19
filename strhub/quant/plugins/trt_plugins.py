import os
import sys
import ctypes
import struct
from typing import List, Optional
import torch
import torch.nn as nn
import tensorrt as trt

# Load compiled CUDA DLL
_DLL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parseq_plugins.dll")
_DLL = None

def get_plugin_dll():
    global _DLL
    if _DLL is None:
        if not os.path.exists(_DLL_PATH):
            raise FileNotFoundError(f"Plugin DLL not found at {_DLL_PATH}. Compile with nvcc first.")
        _DLL = ctypes.CDLL(_DLL_PATH)
    return _DLL


# ==============================================================================
# PyTorch Functions & Symbolics for ONNX Export
# ==============================================================================
class INTFlashAttentionPluginOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
        if q.is_cuda:
            dll = get_plugin_dll()
            out = torch.empty_like(q)
            B, H, N, D = q.shape
            S = k.shape[2]
            stream = torch.cuda.current_stream()
            dll.run_int_flash_attention(
                ctypes.c_void_p(out.data_ptr()),
                ctypes.c_void_p(q.data_ptr()),
                ctypes.c_void_p(k.data_ptr()),
                ctypes.c_void_p(v.data_ptr()),
                ctypes.c_int(B), ctypes.c_int(H), ctypes.c_int(N), ctypes.c_int(S), ctypes.c_int(D),
                ctypes.c_float(scale),
                ctypes.c_void_p(stream.cuda_stream)
            )
            return out
        from strhub.quant.int_flashattention import int_flashattention_forward
        return int_flashattention_forward(q, k, v, scale=scale)

    @staticmethod
    def symbolic(g, q, k, v, scale):
        return g.op("trt.plugins::INTFlashAttentionPlugin", q, k, v, scale_f=float(scale))


class SageAttentionPluginOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float, mode: int = 0) -> torch.Tensor:
        if q.is_cuda:
            dll = get_plugin_dll()
            out = torch.empty_like(q)
            B, H, N, D = q.shape
            S = k.shape[2]
            stream = torch.cuda.current_stream()
            dll.run_sage_attention(
                ctypes.c_void_p(out.data_ptr()),
                ctypes.c_void_p(q.data_ptr()),
                ctypes.c_void_p(k.data_ptr()),
                ctypes.c_void_p(v.data_ptr()),
                ctypes.c_int(B), ctypes.c_int(H), ctypes.c_int(N), ctypes.c_int(S), ctypes.c_int(D),
                ctypes.c_float(scale),
                ctypes.c_int(mode),
                ctypes.c_void_p(stream.cuda_stream)
            )
            return out
        from strhub.quant.sage_attention import sage_attention_forward
        return sage_attention_forward(q, k, v, scale=scale, mode="sageattn_b" if mode == 0 else "sageattn_vb")

    @staticmethod
    def symbolic(g, q, k, v, scale, mode=0):
        return g.op("trt.plugins::SageAttentionPlugin", q, k, v, scale_f=float(scale), mode_i=int(mode))


class IntegerLayerNormPluginOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        if x.is_cuda:
            dll = get_plugin_dll()
            out = torch.empty_like(x)
            D = x.shape[-1]
            num_rows = x.numel() // D
            stream = torch.cuda.current_stream()
            dll.run_integer_layernorm(
                ctypes.c_void_p(out.data_ptr()),
                ctypes.c_void_p(x.data_ptr()),
                None, None,
                ctypes.c_int(num_rows), ctypes.c_int(D), ctypes.c_float(eps),
                ctypes.c_void_p(stream.cuda_stream)
            )
            return out
        # Pure functional CPU fallback
        mean = x.mean(dim=-1, keepdim=True)
        diff = x - mean
        variance = (diff ** 2).mean(dim=-1, keepdim=True)
        n_clamped = torch.clamp(variance + eps, min=1e-8)
        xi = torch.clamp(torch.sqrt(n_clamped), min=1.0)
        for _ in range(4):
            xi = 0.5 * (xi + n_clamped / xi)
        return diff / xi

    @staticmethod
    def symbolic(g, x, eps=1e-5):
        return g.op("trt.plugins::IntegerLayerNormPlugin", x, eps_f=float(eps))


class IntegerGELUPluginOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            dll = get_plugin_dll()
            out = torch.empty_like(x)
            stream = torch.cuda.current_stream()
            dll.run_integer_gelu(
                ctypes.c_void_p(out.data_ptr()),
                ctypes.c_void_p(x.data_ptr()),
                ctypes.c_int(x.numel()),
                ctypes.c_void_p(stream.cuda_stream)
            )
            return out
        # Pure functional CPU fallback
        x_scaled = x * 0.7071067811865475
        sign = torch.sign(x_scaled)
        abs_x = torch.abs(x_scaled)
        clipped = torch.clamp(abs_x, max=1.769)
        poly = -0.2888 * (clipped - 1.769) ** 2 + 1.0
        return 0.5 * x * (1.0 + sign * poly)

    @staticmethod
    def symbolic(g, x):
        return g.op("trt.plugins::IntegerGELUPlugin", x)


class IntegerSoftmaxPluginOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            dll = get_plugin_dll()
            out = torch.empty_like(x)
            N = x.shape[-1]
            num_rows = x.numel() // N
            stream = torch.cuda.current_stream()
            dll.run_integer_softmax(
                ctypes.c_void_p(out.data_ptr()),
                ctypes.c_void_p(x.data_ptr()),
                ctypes.c_int(num_rows), ctypes.c_int(N),
                ctypes.c_void_p(stream.cuda_stream)
            )
            return out
        # Pure functional CPU fallback
        max_val = torch.max(x, dim=-1, keepdim=True)[0]
        x_shifted = x - max_val
        ln2 = 0.6931471805599453
        z = torch.floor(-x_shifted / ln2)
        p = x_shifted + z * ln2
        poly = 0.3585 * (p + 1.353) ** 2 + 0.344
        exp_approx = poly * torch.pow(2.0, -z)
        return exp_approx / torch.clamp(torch.sum(exp_approx, dim=-1, keepdim=True), min=1e-8)

    @staticmethod
    def symbolic(g, x):
        return g.op("trt.plugins::IntegerSoftmaxPlugin", x)


# ==============================================================================
# PyTorch Module Wrappers using the Plugin Ops
# ==============================================================================
class IntegerLayerNormPluginWrapper(nn.Module):
    def __init__(self, normalized_shape, eps: float = 1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            self.normalized_shape = (normalized_shape,)
        else:
            self.normalized_shape = tuple(normalized_shape)
        dim = self.normalized_shape[0]
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = IntegerLayerNormPluginOp.apply(x, self.eps)
        return x_norm * self.weight + self.bias


class IntegerGELUPluginWrapper(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        return IntegerGELUPluginOp.apply(x)


class IntegerSoftmaxPluginWrapper(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return IntegerSoftmaxPluginOp.apply(x)


# ==============================================================================
# 1. INT-FlashAttention TensorRT Plugin
# ==============================================================================
class INTFlashAttentionPluginDynamic(trt.IPluginV2DynamicExt):
    """TensorRT Plugin for fused INT-FlashAttention (Algorithm 1 from arXiv:2409.16997v2).
    Inputs:
      - 0: Q (Float32, shape [B, H, N, D])
      - 1: K (Float32, shape [B, H, S, D])
      - 2: V (Float32, shape [B, H, S, D])
    Outputs:
      - 0: Out (Float32, shape [B, H, N, D])
    """
    def __init__(self, scale: float = 0.125):
        super().__init__()
        self.plugin_type = "INTFlashAttentionPlugin"
        self.plugin_version = "1"
        self.plugin_namespace = ""
        self.num_outputs = 1
        self.scale = float(scale)

    def get_output_datatype(self, index: int, input_types: List[trt.DataType]) -> trt.DataType:
        return input_types[0]

    def get_output_dimensions(self, output_index: int, inputs: List[trt.DimsExprs], expr_builder: trt.IExprBuilder) -> trt.DimsExprs:
        return inputs[0]

    def supports_format_combination(self, pos: int, in_out: List[trt.PluginTensorDesc], num_inputs: int) -> bool:
        desc = in_out[pos]
        return desc.format == trt.TensorFormat.LINEAR and desc.type in [trt.DataType.FLOAT, trt.DataType.HALF]

    def configure_plugin(self, in_desc: List[trt.DynamicPluginTensorDesc], out_desc: List[trt.DynamicPluginTensorDesc]):
        pass

    def get_workspace_size(self, in_desc: List[trt.PluginTensorDesc], out_desc: List[trt.PluginTensorDesc]) -> int:
        return 0

    def enqueue(self, input_desc: List[trt.PluginTensorDesc], output_desc: List[trt.PluginTensorDesc],
                inputs: List[int], outputs: List[int], workspace: int, stream: int) -> int:
        dll = get_plugin_dll()
        dims_q = input_desc[0].dims
        B, H, N, D = dims_q[0], dims_q[1], dims_q[2], dims_q[3]
        dims_k = input_desc[1].dims
        S = dims_k[2]

        q_ptr = ctypes.c_void_p(inputs[0])
        k_ptr = ctypes.c_void_p(inputs[1])
        v_ptr = ctypes.c_void_p(inputs[2])
        out_ptr = ctypes.c_void_p(outputs[0])
        stream_ptr = ctypes.c_void_p(stream)

        dll.run_int_flash_attention(
            out_ptr, q_ptr, k_ptr, v_ptr,
            ctypes.c_int(B), ctypes.c_int(H), ctypes.c_int(N), ctypes.c_int(S), ctypes.c_int(D),
            ctypes.c_float(self.scale),
            stream_ptr
        )
        return 0

    def clone(self) -> "INTFlashAttentionPluginDynamic":
        return INTFlashAttentionPluginDynamic(scale=self.scale)

    def get_serialization_size(self) -> int:
        return struct.calcsize("f")

    def serialize(self) -> bytes:
        return struct.pack("f", self.scale)


class INTFlashAttentionPluginCreator(trt.IPluginCreator):
    def __init__(self):
        super().__init__()
        self.name = "INTFlashAttentionPlugin"
        self.plugin_version = "1"
        self.plugin_namespace = ""
        self.field_names = trt.PluginFieldCollection()

    def create_plugin(self, name: str, field_collection: trt.PluginFieldCollection_) -> trt.IPluginV2:
        scale = 0.125
        for field in field_collection:
            if field.name == "scale":
                scale = float(field.data[0]) if hasattr(field.data, "__getitem__") else float(field.data)
        return INTFlashAttentionPluginDynamic(scale=scale)

    def deserialize_plugin(self, name: str, serialized_plugin: bytes) -> trt.IPluginV2:
        if len(serialized_plugin) >= struct.calcsize("f"):
            scale = struct.unpack("f", serialized_plugin[:struct.calcsize("f")])[0]
        else:
            scale = 0.125
        return INTFlashAttentionPluginDynamic(scale=scale)


class SageAttentionPluginDynamic(trt.IPluginV2DynamicExt):
    """TensorRT Plugin for fused SageAttention (Algorithm 1 from arXiv:2410.02367v9).
    Features token-averaged key smoothing, INT8 x INT8 -> INT32 GEMM, and online softmax.
    Inputs:
      - 0: Q (Float32, shape [B, H, N, D])
      - 1: K (Float32, shape [B, H, S, D])
      - 2: V (Float32, shape [B, H, S, D])
    Outputs:
      - 0: Out (Float32, shape [B, H, N, D])
    """
    def __init__(self, scale: float = 0.125, mode: int = 0):
        super().__init__()
        self.plugin_type = "SageAttentionPlugin"
        self.plugin_version = "1"
        self.plugin_namespace = ""
        self.num_outputs = 1
        self.scale = float(scale)
        self.mode = int(mode)

    def get_output_datatype(self, index: int, input_types: List[trt.DataType]) -> trt.DataType:
        return input_types[0]

    def get_output_dimensions(self, output_index: int, inputs: List[trt.DimsExprs], expr_builder: trt.IExprBuilder) -> trt.DimsExprs:
        return inputs[0]

    def supports_format_combination(self, pos: int, in_out: List[trt.PluginTensorDesc], num_inputs: int) -> bool:
        desc = in_out[pos]
        return desc.format == trt.TensorFormat.LINEAR and desc.type in [trt.DataType.FLOAT, trt.DataType.HALF]

    def configure_plugin(self, in_desc: List[trt.DynamicPluginTensorDesc], out_desc: List[trt.DynamicPluginTensorDesc]):
        pass

    def get_workspace_size(self, in_desc: List[trt.PluginTensorDesc], out_desc: List[trt.PluginTensorDesc]) -> int:
        return 0

    def enqueue(self, input_desc: List[trt.PluginTensorDesc], output_desc: List[trt.PluginTensorDesc],
                inputs: List[int], outputs: List[int], workspace: int, stream: int) -> int:
        dll = get_plugin_dll()
        dims_q = input_desc[0].dims
        B, H, N, D = dims_q[0], dims_q[1], dims_q[2], dims_q[3]
        dims_k = input_desc[1].dims
        S = dims_k[2]

        q_ptr = ctypes.c_void_p(inputs[0])
        k_ptr = ctypes.c_void_p(inputs[1])
        v_ptr = ctypes.c_void_p(inputs[2])
        out_ptr = ctypes.c_void_p(outputs[0])
        stream_ptr = ctypes.c_void_p(stream)

        dll.run_sage_attention(
            out_ptr, q_ptr, k_ptr, v_ptr,
            ctypes.c_int(B), ctypes.c_int(H), ctypes.c_int(N), ctypes.c_int(S), ctypes.c_int(D),
            ctypes.c_float(self.scale),
            ctypes.c_int(self.mode),
            stream_ptr
        )
        return 0

    def clone(self) -> "SageAttentionPluginDynamic":
        return SageAttentionPluginDynamic(scale=self.scale, mode=self.mode)

    def get_serialization_size(self) -> int:
        return struct.calcsize("fi")

    def serialize(self) -> bytes:
        return struct.pack("fi", self.scale, self.mode)


class SageAttentionPluginCreator(trt.IPluginCreator):
    def __init__(self):
        super().__init__()
        self.name = "SageAttentionPlugin"
        self.plugin_version = "1"
        self.plugin_namespace = ""
        self.field_names = trt.PluginFieldCollection()

    def create_plugin(self, name: str, field_collection: trt.PluginFieldCollection_) -> trt.IPluginV2:
        scale = 0.125
        mode = 0
        for field in field_collection:
            if field.name == "scale":
                scale = float(field.data[0]) if hasattr(field.data, "__getitem__") else float(field.data)
            elif field.name == "mode":
                mode = int(field.data[0]) if hasattr(field.data, "__getitem__") else int(field.data)
        return SageAttentionPluginDynamic(scale=scale, mode=mode)

    def deserialize_plugin(self, name: str, serialized_plugin: bytes) -> trt.IPluginV2:
        if len(serialized_plugin) >= struct.calcsize("fi"):
            scale, mode = struct.unpack("fi", serialized_plugin[:struct.calcsize("fi")])
        elif len(serialized_plugin) >= struct.calcsize("f"):
            scale = struct.unpack("f", serialized_plugin[:struct.calcsize("f")])[0]
            mode = 0
        else:
            scale = 0.125
            mode = 0
        return SageAttentionPluginDynamic(scale=scale, mode=mode)


# ==============================================================================
# 2. Integer LayerNorm Plugin
# ==============================================================================
class IntegerLayerNormPluginDynamic(trt.IPluginV2DynamicExt):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.plugin_type = "IntegerLayerNormPlugin"
        self.plugin_version = "1"
        self.plugin_namespace = ""
        self.num_outputs = 1
        self.eps = float(eps)

    def get_output_datatype(self, index: int, input_types: List[trt.DataType]) -> trt.DataType:
        return input_types[0]

    def get_output_dimensions(self, output_index: int, inputs: List[trt.DimsExprs], expr_builder: trt.IExprBuilder) -> trt.DimsExprs:
        return inputs[0]

    def supports_format_combination(self, pos: int, in_out: List[trt.PluginTensorDesc], num_inputs: int) -> bool:
        desc = in_out[pos]
        return desc.format == trt.TensorFormat.LINEAR and desc.type == trt.DataType.FLOAT

    def configure_plugin(self, in_desc: List[trt.DynamicPluginTensorDesc], out_desc: List[trt.DynamicPluginTensorDesc]):
        pass

    def get_workspace_size(self, in_desc: List[trt.PluginTensorDesc], out_desc: List[trt.PluginTensorDesc]) -> int:
        return 0

    def enqueue(self, input_desc: List[trt.PluginTensorDesc], output_desc: List[trt.PluginTensorDesc],
                inputs: List[int], outputs: List[int], workspace: int, stream: int) -> int:
        dll = get_plugin_dll()
        dims = input_desc[0].dims
        D = dims[-1]
        total_elements = 1
        for d in dims:
            total_elements *= d
        num_rows = total_elements // D

        in_ptr = ctypes.c_void_p(inputs[0])
        out_ptr = ctypes.c_void_p(outputs[0])
        stream_ptr = ctypes.c_void_p(stream)

        dll.run_integer_layernorm(
            out_ptr, in_ptr, None, None,
            ctypes.c_int(num_rows), ctypes.c_int(D), ctypes.c_float(self.eps),
            stream_ptr
        )
        return 0

    def clone(self) -> "IntegerLayerNormPluginDynamic":
        return IntegerLayerNormPluginDynamic(eps=self.eps)

    def get_serialization_size(self) -> int:
        return struct.calcsize("f")

    def serialize(self) -> bytes:
        return struct.pack("f", self.eps)


class IntegerLayerNormPluginCreator(trt.IPluginCreator):
    def __init__(self):
        super().__init__()
        self.name = "IntegerLayerNormPlugin"
        self.plugin_version = "1"
        self.plugin_namespace = ""
        self.field_names = trt.PluginFieldCollection()

    def create_plugin(self, name: str, field_collection: trt.PluginFieldCollection_) -> trt.IPluginV2:
        eps = 1e-5
        for field in field_collection:
            if field.name == "eps":
                eps = float(field.data[0]) if hasattr(field.data, "__getitem__") else float(field.data)
        return IntegerLayerNormPluginDynamic(eps=eps)

    def deserialize_plugin(self, name: str, serialized_plugin: bytes) -> trt.IPluginV2:
        eps = struct.unpack("f", serialized_plugin[:struct.calcsize("f")])[0] if len(serialized_plugin) >= 4 else 1e-5
        return IntegerLayerNormPluginDynamic(eps=eps)


# ==============================================================================
# 3. Integer GELU Plugin
# ==============================================================================
class IntegerGELUPluginDynamic(trt.IPluginV2DynamicExt):
    def __init__(self):
        super().__init__()
        self.plugin_type = "IntegerGELUPlugin"
        self.plugin_version = "1"
        self.plugin_namespace = ""
        self.num_outputs = 1

    def get_output_datatype(self, index: int, input_types: List[trt.DataType]) -> trt.DataType:
        return input_types[0]

    def get_output_dimensions(self, output_index: int, inputs: List[trt.DimsExprs], expr_builder: trt.IExprBuilder) -> trt.DimsExprs:
        return inputs[0]

    def supports_format_combination(self, pos: int, in_out: List[trt.PluginTensorDesc], num_inputs: int) -> bool:
        desc = in_out[pos]
        return desc.format == trt.TensorFormat.LINEAR and desc.type == trt.DataType.FLOAT

    def configure_plugin(self, in_desc: List[trt.DynamicPluginTensorDesc], out_desc: List[trt.DynamicPluginTensorDesc]):
        pass

    def get_workspace_size(self, in_desc: List[trt.PluginTensorDesc], out_desc: List[trt.PluginTensorDesc]) -> int:
        return 0

    def enqueue(self, input_desc: List[trt.PluginTensorDesc], output_desc: List[trt.PluginTensorDesc],
                inputs: List[int], outputs: List[int], workspace: int, stream: int) -> int:
        dll = get_plugin_dll()
        dims = input_desc[0].dims
        total_elements = 1
        for d in dims:
            total_elements *= d

        in_ptr = ctypes.c_void_p(inputs[0])
        out_ptr = ctypes.c_void_p(outputs[0])
        stream_ptr = ctypes.c_void_p(stream)

        dll.run_integer_gelu(
            out_ptr, in_ptr, ctypes.c_int(total_elements), stream_ptr
        )
        return 0

    def clone(self) -> "IntegerGELUPluginDynamic":
        return IntegerGELUPluginDynamic()

    def get_serialization_size(self) -> int:
        return 0

    def serialize(self) -> bytes:
        return b""


class IntegerGELUPluginCreator(trt.IPluginCreator):
    def __init__(self):
        super().__init__()
        self.name = "IntegerGELUPlugin"
        self.plugin_version = "1"
        self.plugin_namespace = ""
        self.field_names = trt.PluginFieldCollection()

    def create_plugin(self, name: str, field_collection: trt.PluginFieldCollection_) -> trt.IPluginV2:
        return IntegerGELUPluginDynamic()

    def deserialize_plugin(self, name: str, serialized_plugin: bytes) -> trt.IPluginV2:
        return IntegerGELUPluginDynamic()


# ==============================================================================
# 4. Integer Softmax Plugin
# ==============================================================================
class IntegerSoftmaxPluginDynamic(trt.IPluginV2DynamicExt):
    def __init__(self):
        super().__init__()
        self.plugin_type = "IntegerSoftmaxPlugin"
        self.plugin_version = "1"
        self.plugin_namespace = ""
        self.num_outputs = 1

    def get_output_datatype(self, index: int, input_types: List[trt.DataType]) -> trt.DataType:
        return input_types[0]

    def get_output_dimensions(self, output_index: int, inputs: List[trt.DimsExprs], expr_builder: trt.IExprBuilder) -> trt.DimsExprs:
        return inputs[0]

    def supports_format_combination(self, pos: int, in_out: List[trt.PluginTensorDesc], num_inputs: int) -> bool:
        desc = in_out[pos]
        return desc.format == trt.TensorFormat.LINEAR and desc.type == trt.DataType.FLOAT

    def configure_plugin(self, in_desc: List[trt.DynamicPluginTensorDesc], out_desc: List[trt.DynamicPluginTensorDesc]):
        pass

    def get_workspace_size(self, in_desc: List[trt.PluginTensorDesc], out_desc: List[trt.PluginTensorDesc]) -> int:
        return 0

    def enqueue(self, input_desc: List[trt.PluginTensorDesc], output_desc: List[trt.PluginTensorDesc],
                inputs: List[int], outputs: List[int], workspace: int, stream: int) -> int:
        dll = get_plugin_dll()
        dims = input_desc[0].dims
        N = dims[-1]
        total_elements = 1
        for d in dims:
            total_elements *= d
        num_rows = total_elements // N

        in_ptr = ctypes.c_void_p(inputs[0])
        out_ptr = ctypes.c_void_p(outputs[0])
        stream_ptr = ctypes.c_void_p(stream)

        dll.run_integer_softmax(
            out_ptr, in_ptr, ctypes.c_int(num_rows), ctypes.c_int(N), stream_ptr
        )
        return 0

    def clone(self) -> "IntegerSoftmaxPluginDynamic":
        return IntegerSoftmaxPluginDynamic()

    def get_serialization_size(self) -> int:
        return 0

    def serialize(self) -> bytes:
        return b""


class IntegerSoftmaxPluginCreator(trt.IPluginCreator):
    def __init__(self):
        super().__init__()
        self.name = "IntegerSoftmaxPlugin"
        self.plugin_version = "1"
        self.plugin_namespace = ""
        self.field_names = trt.PluginFieldCollection()

    def create_plugin(self, name: str, field_collection: trt.PluginFieldCollection_) -> trt.IPluginV2:
        return IntegerSoftmaxPluginDynamic()

    def deserialize_plugin(self, name: str, serialized_plugin: bytes) -> trt.IPluginV2:
        return IntegerSoftmaxPluginDynamic()


# ==============================================================================
# Global Registration
# ==============================================================================
_REGISTERED = False

def register_parseq_plugins():
    """Registers all custom PARSeq plugins into TensorRT's plugin registry."""
    global _REGISTERED
    if _REGISTERED:
        return
    registry = trt.get_plugin_registry()
    existing_creators = {c.name for c in registry.all_creators}

    creator_classes = [
        INTFlashAttentionPluginCreator,
        SageAttentionPluginCreator,
        IntegerLayerNormPluginCreator,
        IntegerGELUPluginCreator,
        IntegerSoftmaxPluginCreator,
    ]

    for cls in creator_classes:
        for ns in ["", "trt.plugins"]:
            try:
                creator = cls()
                creator.plugin_namespace = ns
                registry.register_creator(creator, ns)
            except Exception:
                pass
        dummy = cls()
        print(f"[TRT Plugins] Registered creator: {dummy.name} v{dummy.plugin_version}")

    _REGISTERED = True
