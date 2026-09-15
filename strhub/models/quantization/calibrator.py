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
