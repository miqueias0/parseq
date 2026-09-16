# Scene Text Recognition Model Hub - Activation Calibrator
# Collects activation distributions and channel-wise outlier profiles for INT8 & SmoothQuant

from typing import Dict, List, Optional
import torch
import torch.nn as nn


class ActivationCalibrator:
    """Forward hook calibrator for capturing layer activation statistics in PARSeq.
    
    Collects:
        - ch_max: Maximum absolute value per channel across calibration batches.
        - tensor_max: Overall maximum absolute activation for the layer.
        - count: Total samples observed.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks = []
        self.stats: Dict[str, Dict[str, torch.Tensor]] = {}
        self.is_calibrating = False

    def _hook_fn(self, name: str):
        def fn(module, input, output):
            if not self.is_calibrating:
                return
            with torch.no_grad():
                # For activations, inspect the output of LayerNorm or input to Linear
                tensor = output if isinstance(output, torch.Tensor) else output[0]
                if not isinstance(tensor, torch.Tensor):
                    return
                # Shape is typically [B, N, C] or [B, C, H, W]
                t = tensor.detach().float()
                if t.dim() == 3: # [B, N, C]
                    ch_max = torch.amax(torch.abs(t), dim=(0, 1)) # [C]
                elif t.dim() == 2: # [B, C]
                    ch_max = torch.amax(torch.abs(t), dim=0) # [C]
                elif t.dim() == 4: # [B, C, H, W]
                    ch_max = torch.amax(torch.abs(t), dim=(0, 2, 3)) # [C]
                else:
                    ch_max = torch.amax(torch.abs(t))

                t_max = torch.max(torch.abs(t))

                if name not in self.stats:
                    self.stats[name] = {
                        "ch_max": ch_max.cpu(),
                        "tensor_max": t_max.cpu(),
                        "count": torch.tensor(t.shape[0]),
                    }
                else:
                    self.stats[name]["ch_max"] = torch.maximum(self.stats[name]["ch_max"], ch_max.cpu())
                    self.stats[name]["tensor_max"] = torch.maximum(self.stats[name]["tensor_max"], t_max.cpu())
                    self.stats[name]["count"] += t.shape[0]

        return fn

    def start(self, target_modules: Optional[List[type]] = None):
        """Attaches hooks to the model."""
        if target_modules is None:
            target_modules = [nn.LayerNorm, nn.Linear]

        self.stop()
        self.stats.clear()
        self.is_calibrating = True

        for name, module in self.model.named_modules():
            if any(isinstance(module, t) for t in target_modules):
                hook = module.register_forward_hook(self._hook_fn(name))
                self.hooks.append(hook)

    def stop(self):
        """Removes hooks from the model."""
        self.is_calibrating = False
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def get_layer_stats(self, name: str) -> Optional[Dict[str, torch.Tensor]]:
        return self.stats.get(name, None)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class PARSeqCalibrationDataReader:
    """Feeds representative input batches for ONNX Runtime static calibration & QDQ generation.
    
    Adheres to ONNX Runtime CalibrationDataReader specification:
    Supplies batches of preprocessed image tensors [B, 3, H, W] to compute optimal
    activation quantization scaling factors and zero-points for TensorRT.
    """

    def __init__(
        self,
        data_source: Optional[object] = None,
        input_name: str = "images",
        batch_size: int = 1,
        img_size: tuple = (32, 128),
        max_samples: int = 64,
    ):
        self.input_name = input_name
        self.batch_size = batch_size
        self.img_size = img_size
        self.max_samples = max_samples
        self.batches = []
        self._current_idx = 0

        self._build_batches(data_source)

    def _build_batches(self, data_source):
        import numpy as np
        from pathlib import Path

        # 1. PyTorch DataLoader
        if hasattr(data_source, "__iter__") and hasattr(data_source, "dataset"):
            count = 0
            for batch in data_source:
                imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
                if hasattr(imgs, "cpu"):
                    arr = imgs.cpu().numpy().astype(np.float32)
                else:
                    arr = np.array(imgs, dtype=np.float32)
                self.batches.append(arr)
                count += arr.shape[0]
                if count >= self.max_samples:
                    break
            return

        # 2. Image directory or list of file paths
        img_paths = []
        if isinstance(data_source, (str, Path)):
            p = Path(data_source)
            if p.is_dir():
                for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                    img_paths.extend(p.glob(ext))
            elif p.is_file():
                img_paths.append(p)
        elif isinstance(data_source, (list, tuple)):
            img_paths = [Path(f) for f in data_source if Path(f).exists()]

        img_paths = sorted(img_paths)[:self.max_samples]

        if img_paths:
            try:
                from PIL import Image
                current_batch = []
                for ip in img_paths:
                    try:
                        with Image.open(ip) as img:
                            img = img.convert("RGB").resize((self.img_size[1], self.img_size[0]))
                            arr = np.array(img, dtype=np.float32) / 255.0
                            arr = (arr - 0.5) / 0.5
                            arr = np.transpose(arr, (2, 0, 1))
                            current_batch.append(arr)
                            if len(current_batch) == self.batch_size:
                                self.batches.append(np.stack(current_batch, axis=0))
                                current_batch = []
                    except Exception:
                        continue
                if current_batch:
                    self.batches.append(np.stack(current_batch, axis=0))
            except ImportError:
                pass

        # 3. Synthetic representative distribution fallback
        if not self.batches:
            num_batches = max(1, self.max_samples // self.batch_size)
            for _ in range(num_batches):
                synth = np.random.randn(self.batch_size, 3, self.img_size[0], self.img_size[1]).astype(np.float32)
                synth = np.clip(synth * 0.5, -1.0, 1.0)
                self.batches.append(synth)

    def get_next(self) -> Optional[dict]:
        if self._current_idx < len(self.batches):
            batch = self.batches[self._current_idx]
            self._current_idx += 1
            return {self.input_name: batch}
        return None

    def rewind(self):
        self._current_idx = 0

