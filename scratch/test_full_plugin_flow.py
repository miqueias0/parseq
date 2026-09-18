import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
import tensorrt as trt
from strhub.quant.plugins.trt_plugins import register_parseq_plugins, get_plugin_dll
import ctypes

# Register plugins
register_parseq_plugins()

class INTFlashAttentionPluginOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale):
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

    @staticmethod
    def symbolic(g, q, k, v, scale):
        return g.op("INTFlashAttentionPlugin", q, k, v, scale_f=scale)

class MiniAttentionModel(nn.Module):
    def __init__(self, H=6, D=64):
        super().__init__()
        self.H = H
        self.D = D
        self.scale = 1.0 / (D ** 0.5)

    def forward(self, q, k, v):
        return INTFlashAttentionPluginOp.apply(q, k, v, self.scale)

model = MiniAttentionModel().cuda().eval()
B, H, N, S, D = 1, 6, 217, 217, 64
q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float32)
v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float32)

onnx_path = "scratch/mini_int_fa.onnx"
torch.onnx.export(
    model,
    (q, k, v),
    onnx_path,
    input_names=["q", "k", "v"],
    output_names=["out"],
    opset_version=18,
    dynamo=False
)
print("Mini model exported to ONNX.")

# Build TensorRT Engine
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network()
parser = trt.OnnxParser(network, logger)

with open(onnx_path, "rb") as f:
    success = parser.parse(f.read())

print("ONNX parsed:", success)
if not success:
    for i in range(parser.num_errors):
        print(parser.get_error(i))
    sys.exit(1)

config = builder.create_builder_config()
plan = builder.build_serialized_network(network, config)
print(f"Plan built: {plan.nbytes} bytes.")

# Run inference
runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(plan)
context = engine.create_execution_context()

out_trt = torch.empty_like(q)
context.set_tensor_address("q", q.data_ptr())
context.set_tensor_address("k", k.data_ptr())
context.set_tensor_address("v", v.data_ptr())
context.set_tensor_address("out", out_trt.data_ptr())

stream = torch.cuda.current_stream()
context.execute_async_v3(stream.cuda_stream)
torch.cuda.synchronize()

print("TensorRT inference complete! Output shape:", out_trt.shape, "mean:", out_trt.mean().item())

# Compare TRT execution with PyTorch plugin forward
out_py = model(q, k, v)
diff = (out_trt - out_py).abs().max().item()
print(f"TRT vs PyTorch Max Abs Diff: {diff:.6f}")
