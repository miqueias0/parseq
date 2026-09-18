import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
import onnx
from onnx import helper, TensorProto
import tensorrt as trt

# 1. Custom PyTorch Function & Symbolic
class CustomDoublePluginOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x * 2.0

    @staticmethod
    def symbolic(g, x):
        return g.op("CustomDoublePlugin", x)

class ModelWithPlugin(nn.Module):
    def forward(self, x):
        return CustomDoublePluginOp.apply(x)

m = ModelWithPlugin().cuda().eval()
x = torch.randn(1, 16, device="cuda")

torch.onnx.export(
    m,
    x,
    "scratch/test_custom_plugin.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=18,
    dynamo=False
)

# 2. Register Plugin in TRT
class DoublePlugin(trt.IPluginV2DynamicExt):
    def __init__(self):
        super().__init__()
        self.plugin_type = "CustomDoublePlugin"
        self.plugin_version = "1"
        self.plugin_namespace = ""
        self.num_outputs = 1

    def get_output_datatype(self, index, input_types):
        return input_types[0]

    def get_output_dimensions(self, output_index, inputs, expr_builder):
        return inputs[0]

    def supports_format_combination(self, pos, in_out, num_inputs):
        return in_out[pos].format == trt.TensorFormat.LINEAR and in_out[pos].type == trt.DataType.FLOAT

    def configure_plugin(self, in_desc, out_desc): pass
    def get_workspace_size(self, in_desc, out_desc): return 0

    def enqueue(self, input_desc, output_desc, inputs, outputs, workspace, stream):
        return 0

    def clone(self): return DoublePlugin()
    def get_serialization_size(self): return 0
    def serialize(self): return b""

class DoublePluginCreator(trt.IPluginCreator):
    def __init__(self):
        super().__init__()
        self.name = "CustomDoublePlugin"
        self.plugin_version = "1"
        self.plugin_namespace = ""
        self.field_names = trt.PluginFieldCollection()

    def create_plugin(self, name, field_collection):
        print(f"DoublePluginCreator: created plugin for {name}")
        return DoublePlugin()

    def deserialize_plugin(self, name, serialized_plugin):
        return DoublePlugin()

registry = trt.get_plugin_registry()
registry.register_creator(DoublePluginCreator(), "")

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network()
parser = trt.OnnxParser(network, logger)

with open("scratch/test_custom_plugin.onnx", "rb") as f:
    success = parser.parse(f.read())

print("ONNX Parse status:", success)
if not success:
    for i in range(parser.num_errors):
        print("Parser error:", parser.get_error(i))
else:
    config = builder.create_builder_config()
    plan = builder.build_serialized_network(network, config)
    print("Plan built successfully! Bytes:", plan.nbytes)
