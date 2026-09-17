"""
Evaluation Adapter for Quantized and TensorRT / ONNX Models in test.py
=====================================================================
Allows test.py to seamlessly evaluate:
1. PTQ / Calibrated checkpoints (e.g. parseq_int8_calibrated.pt)
2. QAT fine-tuned checkpoints (e.g. parseq_qat_finetuned.pt)
3. 100% Integer-Only models (via --quant_mode integer_only or auto-detection)
4. TensorRT INT8 engines (e.g. parseq_int8.engine)
5. ONNX Q/DQ models (e.g. parseq_qdq.onnx)
"""

import os
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from strhub.models.utils import create_model, load_from_checkpoint
from .parseq_quantizer import IntegerPARSeq, quantize_parseq


class TRTEncoderWrapper(nn.Module):
    """
    Wraps a compiled TensorRT engine to replace model.encoder.
    Executes image encoding on GPU using native TensorRT context.
    Automatically handles arbitrary batch sizes via dynamic shape execution and slicing.
    """
    def __init__(self, engine_path: str, device: str = "cuda"):
        super().__init__()
        self.engine_path = engine_path
        self.device = device

        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Cannot run TensorRT engine '{engine_path}' because CUDA is not available on this system. "
                "TensorRT engines require an NVIDIA GPU. For CPU evaluation, use the ONNX model or Integer-Only checkpoint."
            )

        import tensorrt as trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Query max batch size supported by profile 0
        try:
            _, _, max_s = self.engine.get_tensor_profile_shape("images", 0)
            self.max_batch_size = max(int(max_s[0]), 1)
        except Exception:
            self.max_batch_size = 256

    def _execute_subbatch(self, images: Tensor) -> Tensor:
        b = images.shape[0]
        self.context.set_input_shape("images", (b, 3, 32, 128))
        out_shape = tuple(self.context.get_tensor_shape("memory"))
        out_shape = (b,) + out_shape[1:]
        memory = torch.empty(out_shape, dtype=torch.float32, device=images.device)

        self.context.set_tensor_address("images", images.data_ptr())
        self.context.set_tensor_address("memory", memory.data_ptr())
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return memory

    def forward(self, images: Tensor) -> Tensor:
        # images: [B, 3, 32, 128] on CUDA
        B = images.shape[0]
        if B <= self.max_batch_size:
            return self._execute_subbatch(images)

        # Chunk across max_batch_size to never exceed optimization profile
        chunks = []
        for i in range(0, B, self.max_batch_size):
            chunk = images[i : i + self.max_batch_size]
            chunks.append(self._execute_subbatch(chunk))
        return torch.cat(chunks, dim=0)


class ONNXEncoderWrapper(nn.Module):
    """
    Wraps an exported ONNX Q/DQ model to replace model.encoder.
    Supports both CPU (VNNI / AVX-512) and GPU (CUDA / TensorrtExecutionProvider).
    """
    def __init__(self, onnx_path: str, device: str = "cpu"):
        super().__init__()
        self.onnx_path = onnx_path
        self.device = device

        import onnxruntime as ort
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def forward(self, images: Tensor) -> Tensor:
        # Convert torch tensor to numpy
        img_np = images.detach().cpu().numpy()
        outputs = self.session.run([self.output_name], {self.input_name: img_np})
        return torch.from_numpy(outputs[0]).to(images.device)


def load_model_for_testing(
    checkpoint_path: str,
    quant_mode: str = "none",
    device: str = "cpu",
    base_checkpoint: Optional[str] = None,
    **kwargs,
) -> nn.Module:
    """
    Unified model loader for test.py supporting:
    - Standard checkpoints / pretrained IDs
    - Calibrated INT8 checkpoints (*.pt)
    - QAT fine-tuned checkpoints (*.pt)
    - TensorRT compiled engines (*.engine / *.plan)
    - ONNX Q/DQ models (*.onnx)
    - On-the-fly quantization via --quant_mode {none, ptq, qat, integer_only}
    """
    def _load_base():
        if base_checkpoint and os.path.isfile(base_checkpoint):
            print(f"[*] Loading base model weights from: {base_checkpoint}")
            try:
                return load_from_checkpoint(base_checkpoint, **kwargs)
            except Exception:
                b_ckpt = torch.load(base_checkpoint, map_location="cpu")
                if isinstance(b_ckpt, dict) and "hyper_parameters" in b_ckpt:
                    from strhub.models.parseq.system import PARSeq as PARSeqSystem
                    hp = dict(b_ckpt["hyper_parameters"])
                    hp.update(kwargs)
                    sys_m = PARSeqSystem(**hp)
                else:
                    sys_m = create_model("parseq", pretrained=False, **kwargs)
                state = b_ckpt.get("state_dict", b_ckpt)
                clean_state = {k.replace("model.", ""): v for k, v in state.items()}
                sys_m.model.load_state_dict(clean_state, strict=False)
                return sys_m
        return create_model("parseq", pretrained=False, **kwargs)

    # 1. TensorRT Engine (.engine / .plan)
    if checkpoint_path.endswith((".engine", ".plan")):
        print(f"[*] Loading base PARSeq system for TensorRT engine evaluation: {checkpoint_path}")
        system = _load_base()
        trt_encoder = TRTEncoderWrapper(checkpoint_path, device=device)
        system.model.encoder = trt_encoder
        return system

    # 2. ONNX Q/DQ Model (.onnx)
    if checkpoint_path.endswith(".onnx"):
        print(f"[*] Loading base PARSeq system for ONNX Q/DQ model evaluation: {checkpoint_path}")
        system = _load_base()
        onnx_encoder = ONNXEncoderWrapper(checkpoint_path, device=device)
        system.model.encoder = onnx_encoder
        return system

    # 3. Checkpoint (.pt) or pretrained ID
    if os.path.isfile(checkpoint_path) and checkpoint_path.endswith(".pt"):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        is_dict = isinstance(ckpt, dict)

        def _get_configured_system():
            if base_checkpoint and os.path.isfile(base_checkpoint):
                return _load_base()
            if is_dict and "hyper_parameters" in ckpt:
                from strhub.models.parseq.system import PARSeq as PARSeqSystem
                hp = dict(ckpt["hyper_parameters"])
                hp.update(kwargs)
                return PARSeqSystem(**hp)
            return create_model("parseq", pretrained=False, **kwargs)

        # Explicit Integer-Only requested
        if quant_mode == "integer_only":
            print(f"[*] Loading 100% Integer-Only model from checkpoint: {checkpoint_path}")
            system = _get_configured_system()
            int_model = IntegerPARSeq(system.model, scale_dict=ckpt.get("scale_dict", {}))
            if "integer_state_dict" in ckpt:
                int_model.load_state_dict(ckpt["integer_state_dict"])
            elif "state_dict" in ckpt:
                int_model.load_state_dict(ckpt["state_dict"], strict=False)
            system.model = int_model
            return system

        # Check if checkpoint is from calibrate_int8.py (has scale_dict or assignments)
        if is_dict and ("scale_dict" in ckpt or "assignments" in ckpt):
            print(f"[*] Loading calibrated INT8 model (QDQ ViT Encoder) from: {checkpoint_path}")
            system = _get_configured_system()
            state = ckpt.get("state_dict", ckpt)
            clean_state = {k.replace("model.", ""): v for k, v in state.items()}
            system.model.load_state_dict(clean_state, strict=False)

            from .trt_exporter import QDQViTEncoder
            system.model.encoder = QDQViTEncoder(
                system.model.encoder,
                state_dict=clean_state,
                scale_dict=ckpt.get("scale_dict", {}),
            )
            return system

        # Check if checkpoint is from train_qat.py
        if is_dict and "state_dict" in ckpt:
            state = ckpt["state_dict"]
            has_qn = any("scale_w" in k or "scale_x" in k for k in state.keys())
            if has_qn or quant_mode == "qat":
                print(f"[*] Loading QAT model from checkpoint: {checkpoint_path}")
                system = _get_configured_system()
                system.model = quantize_parseq(system.model, mode="qat")
                clean_state = {k.replace("model.", ""): v for k, v in state.items()}
                system.model.load_state_dict(clean_state, strict=False)
                return system

        # Standard PyTorch Lightning checkpoint
        try:
            system = load_from_checkpoint(checkpoint_path, **kwargs)
        except Exception:
            system = _get_configured_system()
            state = ckpt.get("state_dict", ckpt)
            clean_state = {k.replace("model.", ""): v for k, v in state.items()}
            system.model.load_state_dict(clean_state, strict=False)
        return system

    else:
        # Standard checkpoint or 'pretrained=<model_id>'
        system = load_from_checkpoint(checkpoint_path, **kwargs)

    # 4. On-the-fly quantization mode requested via --quant_mode
    if quant_mode != "none":
        print(f"[*] Applying on-the-fly quantization mode: '{quant_mode}'")
        system.model = quantize_parseq(system.model, mode=quant_mode)

    return system
