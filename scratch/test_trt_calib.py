import os
import sys
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Ensure paths
sys.path.insert(0, os.path.abspath("."))

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint
from strhub.models.parseq.quantized_parseq import create_model_variant, QuantizedLinear
from tools.export_onnx import ONNXExportWrapper
from tools.build_tensorrt import build_tensorrt_engine
from tools.evaluate_alpr import TensorRTModelWrapper, evaluate_dataset

def main():
    print("Loading system checkpoint...")
    system = load_from_checkpoint("pretrained/parseq_alpr_98.5.ckpt").eval().cuda()
    m5 = create_model_variant("m5", system.model, calibration_file=None).eval().cuda()

    # 1. Collect true input activations on 256 calibration samples
    print("Collecting input activations for all QuantizedLinear modules...")
    hooks = {}
    handles = []
    for name, mod in m5.named_modules():
        if isinstance(mod, QuantizedLinear):
            acts = []
            def make_h(l):
                return lambda m, inp, out: l.append(inp[0].detach().cpu())
            handles.append(mod.register_forward_hook(make_h(acts)))
            hooks[name] = acts

    dm = SceneTextDataModule(
        root_dir="data", train_dir="VeSV_pad",
        img_size=system.hparams.img_size, max_label_length=system.hparams.max_label_length,
        charset_train=system.hparams.charset_train, charset_test=system.hparams.charset_test,
        batch_size=32, num_workers=0, augment=False
    )
    loader = DataLoader(Subset(dm.train_dataset, list(range(256))), batch_size=32, shuffle=False)

    with torch.no_grad():
        for imgs, _ in loader:
            _ = m5(system.tokenizer, imgs.cuda())

    for h in handles:
        h.remove()

    # 2. Compute and set true input scales
    calib_dict = {}
    for name, mod in m5.named_modules():
        if isinstance(mod, QuantizedLinear):
            a_list = hooks[name]
            if len(a_list) > 0:
                max_val = float(max(a.abs().max().item() for a in a_list))
                s = max(max_val / 127.0, 1e-6)
                mod.set_activation_scale(s)
                calib_dict[name] = s
            else:
                mod.set_activation_scale(0.05)
                calib_dict[name] = 0.05

    print(f"Set true input scales on {len(calib_dict)} modules.")

    # 3. Export M5 to ONNX with explicit Q/DQ
    print("Exporting M5 with explicit Q/DQ...")
    m5_cpu = m5.cpu().eval()
    QuantizedLinear.global_export_qdq = True
    wrapper = ONNXExportWrapper(m5_cpu, system.tokenizer).eval()
    dummy = torch.randn(1, 3, 32, 128, dtype=torch.float32)

    os.makedirs("onnx", exist_ok=True)
    onnx_test_path = "onnx/test_m5_calib.onnx"
    torch.onnx.export(
        wrapper,
        dummy,
        onnx_test_path,
        input_names=["images"],
        output_names=["logits"],
        opset_version=18,
        dynamic_axes={"images": {0: "batch_size"}, "logits": {0: "batch_size"}},
        dynamo=False
    )
    QuantizedLinear.global_export_qdq = False
    print(f"ONNX exported successfully: {os.path.getsize(onnx_test_path)/(1024*1024):.2f} MB")

    # 4. Build TensorRT engine
    print("Building TensorRT engine...")
    os.makedirs("trt", exist_ok=True)
    engine_test_path = "trt/test_m5_calib.engine"
    build_tensorrt_engine(
        onnx_path=onnx_test_path,
        engine_path=engine_test_path,
        precision="int8_io",
        max_batch_size=64
    )
    print(f"TRT Engine built: {os.path.getsize(engine_test_path)/(1024*1024):.2f} MB")

    # 5. Evaluate TRT engine on 100 samples
    print("Evaluating TRT engine...")
    trt_model = TensorRTModelWrapper(engine_test_path, device="cuda")
    test_loader = dm.test_dataloaders(["VeSV_pad"])["VeSV_pad"]
    results = evaluate_dataset(trt_model, test_loader, system.tokenizer, system.charset_adapter, torch.device("cuda"), max_samples=100)
    acc = results["exact_plate_accuracy"] * 100.0
    cer = results["character_error_rate"] * 100.0
    print("==========================================")
    print(f"TENSORRT ENGINE EVALUATION RESULTS:")
    print(f"Exact Plate Accuracy: {acc:.2f}%")
    print(f"Character Error Rate: {cer:.2f}%")
    print("==========================================")

if __name__ == "__main__":
    main()
