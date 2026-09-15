# Scene Text Recognition Model Hub - ONNX Export & Runtime Engine
# Copyright 2026 Darwin Bautista / PARSeq ONNX Extensions
#
# Licensed under the Apache License, Version 2.0 (the "License");

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import time
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import onnx
import onnxruntime as ort

from strhub.data.utils import Tokenizer, CharsetAdapter
from strhub.onnx.env import auto_configure_cuda_env


class PARSeqONNXRuntime:
    """High-Performance Inference Engine for PARSeq using ONNX Runtime.

    Supports TensorrtExecutionProvider, CUDAExecutionProvider, and CPUExecutionProvider.
    Includes built-in image preprocessing, inference execution, and token decoding.
    """

    DEFAULT_94 = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    CHARSET_36_UPPER = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    CHARSET_36_LOWER = "0123456789abcdefghijklmnopqrstuvwxyz"

    def __init__(
        self,
        onnx_model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        device_id: int = 0,
        intra_op_threads: int = 4,
        inter_op_threads: int = 1,
        charset_train: Optional[str] = None,
        charset_test: Optional[str] = None,
    ):
        self.onnx_model_path = onnx_model_path
        self.device = device.lower()
        self.device_id = device_id

        # Auto-configure LD_LIBRARY_PATH if running on Linux with CUDA
        if "cuda" in self.device or "tensorrt" in self.device:
            auto_configure_cuda_env(force_reexec=False)
            # Normalize device_id: when CUDA_VISIBLE_DEVICES is used, device 0 is usually the mapped GPU
            if torch.cuda.is_available():
                count = torch.cuda.device_count()
                if self.device_id >= count:
                    self.device_id = 0

        # 1. Parse Metadata from ONNX Model
        onnx_model = onnx.load(onnx_model_path)
        meta = {prop.key: prop.value for prop in onnx_model.metadata_props}
        
        self.img_h = int(meta.get("img_height", "32"))
        self.img_w = int(meta.get("img_width", "128"))
        self.img_size = (self.img_h, self.img_w)
        self.max_label_length = int(meta.get("max_label_length", "25"))
        self.charset_train = charset_train or meta.get("charset_train", self.DEFAULT_94)
        self.charset_test = charset_test or meta.get("charset_test", "0123456789abcdefghijklmnopqrstuvwxyz")

        # 2. Configure Execution Providers
        providers = []
        provider_options = []

        available = ort.get_available_providers()
        if "cuda" in self.device or "tensorrt" in self.device:
            if "TensorrtExecutionProvider" in available and "tensorrt" in self.device:
                providers.append("TensorrtExecutionProvider")
                provider_options.append({"device_id": self.device_id})
            if "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
                provider_options.append({
                    "device_id": self.device_id,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "gpu_mem_limit": 8 * 1024 * 1024 * 1024,
                    "cudnn_conv_algo_search": "DEFAULT",
                    "do_copy_in_default_stream": True,
                })

        # Always add CPU fallback
        providers.append("CPUExecutionProvider")
        provider_options.append({"arena_extend_strategy": "kSameAsRequested"})

        # 3. Session Options
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = intra_op_threads
        sess_opts.inter_op_num_threads = inter_op_threads
        sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_opts.log_severity_level = 3  # Suppress informational memcpy warnings

        self.session = ort.InferenceSession(
            onnx_model_path,
            sess_options=sess_opts,
            providers=providers,
            provider_options=provider_options,
        )

        self.active_provider = self.session.get_providers()[0]
        if ("cuda" in self.device or "tensorrt" in self.device) and self.active_provider == "CPUExecutionProvider":
            print(
                f"[PARSeq ONNX Warning] CUDA was requested, but session loaded with '{self.active_provider}'.\n"
                f"  Tip: To ensure CUDAExecutionProvider finds cuDNN 9 and CUDA 12:\n"
                f"       run 'source setup_cuda_env.sh' or prepend venv nvidia libs to LD_LIBRARY_PATH."
            )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        out_shape = self.session.get_outputs()[0].shape
        # Dynamic sequence length detection (seq_len = max_label_length + 1)
        if len(out_shape) >= 2 and isinstance(out_shape[1], int):
            seq_len = out_shape[1]
            if "max_label_length" not in meta or self.max_label_length != seq_len - 1:
                self.max_label_length = seq_len - 1

        # Dynamic vocabulary detection
        if len(out_shape) >= 3 and isinstance(out_shape[-1], int):
            num_classes = out_shape[-1]
            if num_classes == 95 and len(self.charset_train) != 94:
                self.charset_train = self.DEFAULT_94
                if charset_test is None:
                    self.charset_test = "0123456789abcdefghijklmnopqrstuvwxyz"
            elif num_classes == 37:
                # 36 alphanumeric characters + 1 [EOS]
                meta_cs = meta.get("charset_train", "")
                if charset_train is not None:
                    self.charset_train = charset_train
                elif len(meta_cs) == 36:
                    self.charset_train = meta_cs
                elif len(self.charset_train) != 36:
                    self.charset_train = self.CHARSET_36_UPPER
                if charset_test is None:
                    self.charset_test = self.charset_train

        self.tokenizer = Tokenizer(self.charset_train)
        self.charset_adapter = CharsetAdapter(self.charset_test)

        # Standard Preprocessing Transform
        self.transform = T.Compose([
            T.Resize(self.img_size, T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(0.5, 0.5),
        ])

    def preprocess(self, img: Union[Image.Image, str, np.ndarray, torch.Tensor]) -> np.ndarray:
        """Converts image or tensor into properly normalized NCHW numpy array."""
        input_type = self.session.get_inputs()[0].type
        target_dtype = np.float16 if "float16" in input_type else np.float32

        if isinstance(img, str):
            img = Image.open(img).convert("RGB")

        if isinstance(img, Image.Image):
            tensor = self.transform(img).unsqueeze(0)  # (1, 3, H, W)
            return tensor.numpy().astype(target_dtype)

        elif isinstance(img, torch.Tensor):
            if img.dim() == 3:
                img = img.unsqueeze(0)
            return img.detach().cpu().numpy().astype(target_dtype)

        elif isinstance(img, np.ndarray):
            if img.ndim == 3:
                img = np.expand_dims(img, 0)
            return img.astype(target_dtype)

        else:
            raise TypeError(f"Unsupported image input type: {type(img)}")

    def forward(self, images: Union[np.ndarray, torch.Tensor, Image.Image, str]) -> np.ndarray:
        """Runs inference and returns raw logits of shape (B, L, C)."""
        input_type = self.session.get_inputs()[0].type
        target_dtype = np.float16 if "float16" in input_type else np.float32
        if isinstance(images, np.ndarray) and images.ndim == 4:
            input_data = images.astype(target_dtype) if images.dtype != target_dtype else images
        else:
            input_data = self.preprocess(images)
        outputs = self.session.run([self.output_name], {self.input_name: input_data})
        return outputs[0]

    def decode_logits(self, logits: np.ndarray, cased: bool = True) -> Tuple[List[str], List[float]]:
        """Applies softmax and greedy token decoding to return strings and confidences."""
        # Convert to torch tensor for clean softmax and tokenizer decoding (cast to float32 for softmax stability)
        logits_tensor = torch.from_numpy(logits).float()
        probs = logits_tensor.softmax(-1)
        preds, probs = self.tokenizer.decode(probs)
        
        confidences = []
        cleaned_preds = []
        for pred, prob in zip(preds, probs):
            conf = float(prob.prod().item()) if len(prob) > 0 else 0.0
            confidences.append(conf)
            cleaned_preds.append(pred if cased else self.charset_adapter(pred))

        return cleaned_preds, confidences

    def predict(
        self,
        images: Union[np.ndarray, torch.Tensor, Image.Image, str],
        cased: bool = True,
    ) -> Tuple[List[str], List[float]]:
        """End-to-end prediction returning list of recognized words and confidence scores."""
        logits = self.forward(images)
        return self.decode_logits(logits, cased=cased)

    def benchmark_latency(
        self,
        input_shape: Optional[Tuple[int, int, int, int]] = None,
        warmup_iters: int = 25,
        test_iters: int = 100,
    ) -> Dict[str, float]:
        """Measures inference latency and throughput in ONNX Runtime."""
        if input_shape is None:
            input_shape = (1, 3, self.img_h, self.img_w)
        input_type = self.session.get_inputs()[0].type
        target_dtype = np.float16 if "float16" in input_type else np.float32
        dummy = np.random.randn(*input_shape).astype(target_dtype)

        # Warmup
        for _ in range(warmup_iters):
            _ = self.session.run([self.output_name], {self.input_name: dummy})

        latencies_ms = []
        for _ in range(test_iters):
            t0 = time.perf_counter_ns()
            _ = self.session.run([self.output_name], {self.input_name: dummy})
            t1 = time.perf_counter_ns()
            latencies_ms.append((t1 - t0) / 1_000_000.0)

        arr = np.array(latencies_ms)
        mean_ms = float(np.mean(arr))
        p50_ms = float(np.median(arr))
        p95_ms = float(np.percentile(arr, 95))
        p99_ms = float(np.percentile(arr, 99))
        batch_size = int(input_shape[0]) if input_shape else 1
        fps = (batch_size * 1000.0) / mean_ms if mean_ms > 0 else 0.0

        return {
            "active_provider": self.active_provider,
            "mean_ms": mean_ms,
            "std_ms": float(np.std(arr)),
            "p50_ms": p50_ms,
            "p95_ms": p95_ms,
            "p99_ms": p99_ms,
            "fps": fps,
            "batch_size": input_shape[0],
        }
