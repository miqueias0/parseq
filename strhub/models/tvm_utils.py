"""Apache TVM Integration Utilities for PARSeq / STRHub.
Provides runtime detection, environment guards, and unified executor abstraction
supporting both Relay (GraphExecutor) and Relax (VirtualMachine) compiled modules.
"""

import os
import sys
from typing import Tuple, Any, Optional

# Disable optional torch C DLPack JIT recompilation on TVM import
os.environ["TVM_FFI_DISABLE_TORCH_C_DLPACK"] = "1"

import numpy as np


def check_tvm_available() -> bool:
    """Dynamic guard checking whether Apache TVM runtime is available in the environment."""
    try:
        import tvm
        has_runtime = hasattr(tvm, "runtime")
        has_fe = hasattr(tvm, "relay") or hasattr(tvm, "relax") or hasattr(tvm, "contrib")
        return bool(has_runtime and has_fe)
    except (ImportError, Exception):
        return False


def get_tvm_device(device_str: str = "cuda"):
    """Returns the TVM device descriptor corresponding to device_str."""
    import tvm
    dev_str = str(device_str).lower()
    if "cuda" in dev_str and hasattr(tvm, "cuda") and tvm.cuda().exist:
        device_id = 0
        if ":" in dev_str:
            try:
                device_id = int(dev_str.split(":")[-1])
            except ValueError:
                device_id = 0
        return tvm.cuda(device_id)
    elif "rocm" in dev_str and hasattr(tvm, "rocm") and tvm.rocm().exist:
        return tvm.rocm(0)
    return tvm.cpu(0)


def to_tvm_tensor(arr, dev):
    """Converts a NumPy array or PyTorch tensor into a TVM Tensor / NDArray."""
    import tvm
    import torch

    if isinstance(arr, torch.Tensor):
        arr_np = arr.detach().cpu().numpy()
    else:
        arr_np = np.asarray(arr)

    # 1. Classic TVM / Relay NDArray
    if hasattr(tvm, "nd") and hasattr(tvm.nd, "array"):
        return tvm.nd.array(arr_np, dev)

    # 2. Modern TVM Unity / Relax Tensor
    if hasattr(tvm.runtime, "tensor"):
        return tvm.runtime.tensor(arr_np, dev)

    # 3. Fallback via empty buffer copy
    dtype_str = str(arr_np.dtype)
    t = tvm.runtime.empty(arr_np.shape, dtype_str, dev)
    t.copyfrom(arr_np)
    return t


def tvm_to_numpy(t) -> np.ndarray:
    """Converts a TVM Tensor or NDArray output into a standard NumPy array."""
    if hasattr(t, "numpy"):
        return t.numpy()
    elif hasattr(t, "asnumpy"):
        return t.asnumpy()
    elif isinstance(t, (list, tuple)):
        return tvm_to_numpy(t[0])
    return np.asarray(t)


class TVMRuntimeSession:
    """Unified runtime session executing either a Relay GraphModule or Relax VirtualMachine."""
    def __init__(self, lib_path: str, device: str = "cuda"):
        if not check_tvm_available():
            raise ImportError(
                "Apache TVM runtime is not installed or available. "
                "Please install apache-tvm in your Python environment."
            )
        import tvm

        self.lib_path = lib_path
        self.device_str = device
        self.dev = get_tvm_device(device)
        self.lib = tvm.runtime.load_module(lib_path)
        self.input_name = "images"
        self.is_relax = False
        self.is_relay = False
        self.vm = None
        self.module = None
        self.entry_func = None

        # 1. Check for Relay GraphExecutor format
        implements_default = False
        if hasattr(self.lib, "implements_function"):
            implements_default = self.lib.implements_function("default")
        else:
            try:
                implements_default = self.lib.get_function("default") is not None
            except Exception:
                implements_default = False

        if implements_default:
            try:
                from tvm.contrib import graph_executor
                self.module = graph_executor.GraphModule(self.lib["default"](self.dev))
                self.is_relay = True
            except Exception:
                self.module = None

        # 2. Check for Relax VirtualMachine format
        if not self.is_relay:
            try:
                if hasattr(tvm, "relax") and hasattr(tvm.relax, "VirtualMachine"):
                    self.vm = tvm.relax.VirtualMachine(self.lib, self.dev)
                    self.is_relax = True
            except Exception:
                self.vm = None

        # 3. Direct function entry fallback
        if not self.is_relay and not self.is_relax:
            if hasattr(self.lib, "entry_name") and self.lib.entry_name:
                self.entry_func = self.lib.get_function(self.lib.entry_name)
            else:
                self.entry_func = self.lib

    def run(self, input_data) -> np.ndarray:
        """Executes forward inference on input_data and returns NumPy array of outputs."""
        x_tvm = to_tvm_tensor(input_data, self.dev)

        if self.is_relay and self.module is not None:
            self.module.set_input(self.input_name, x_tvm)
            self.module.run()
            out = self.module.get_output(0)
            return tvm_to_numpy(out)
        elif self.is_relax and self.vm is not None:
            if hasattr(self.vm, "module") and hasattr(self.vm.module, "implements_function") and self.vm.module.implements_function("main"):
                main_fn = self.vm["main"]
            else:
                try:
                    main_fn = self.vm["main"]
                except Exception:
                    main_fn = self.vm
            out = main_fn(x_tvm)
            return tvm_to_numpy(out)
        elif self.entry_func is not None:
            out = self.entry_func(x_tvm)
            return tvm_to_numpy(out)
        else:
            raise RuntimeError("Nenhum mecanismo de execução (GraphModule, VirtualMachine) disponível para este módulo TVM.")
