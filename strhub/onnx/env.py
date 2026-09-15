# Scene Text Recognition Model Hub - ONNX Export & Runtime Engine
# Copyright 2026 Darwin Bautista / PARSeq ONNX Extensions
#
# Licensed under the Apache License, Version 2.0 (the "License");

import os
import sys
import platform
import site
from typing import List


def get_nvidia_lib_paths() -> List[str]:
    """Find all pip-installed nvidia CUDA and cuDNN library directories."""
    paths_to_search = set()
    try:
        for p in site.getsitepackages():
            paths_to_search.add(p)
    except Exception:
        pass
    if hasattr(site, "getusersitepackages"):
        try:
            paths_to_search.add(site.getusersitepackages())
        except Exception:
            pass
    for p in sys.path:
        paths_to_search.add(p)

    nv_lib_paths = []
    for base in paths_to_search:
        nv_dir = os.path.join(base, "nvidia")
        if os.path.isdir(nv_dir):
            for sub in sorted(os.listdir(nv_dir)):
                lib_dir = os.path.join(nv_dir, sub, "lib")
                if os.path.isdir(lib_dir):
                    nv_lib_paths.append(lib_dir)

    return nv_lib_paths


def auto_configure_cuda_env(force_reexec: bool = True) -> bool:
    """Configures LD_LIBRARY_PATH for ONNX Runtime CUDAExecutionProvider.

    pip packages such as nvidia-cudnn-cu12 and nvidia-cublas-cu12 install shared libraries
    inside site-packages/nvidia/*/lib. ONNX Runtime loads libonnxruntime_providers_cuda.so
    via dlopen(), which relies on LD_LIBRARY_PATH at process startup.

    If LD_LIBRARY_PATH does not contain these directories, this function updates
    os.environ["LD_LIBRARY_PATH"] and re-executes the Python interpreter once.

    Returns:
        True if the environment already includes CUDA/cuDNN or was successfully configured.
    """
    if platform.system() != "Linux":
        return True

    env_flag = "_PARSEQ_CUDA_ENV_CONFIGURED"
    if os.environ.get(env_flag) == "1":
        return True

    nv_libs = get_nvidia_lib_paths()
    if not nv_libs:
        return False

    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    current_paths = current_ld.split(":") if current_ld else []

    # Check if any nvidia path is missing from LD_LIBRARY_PATH
    missing_paths = [p for p in nv_libs if p not in current_paths]
    if not missing_paths:
        return True

    new_ld = ":".join(nv_libs) + (":" + current_ld if current_ld else "")
    os.environ["LD_LIBRARY_PATH"] = new_ld
    os.environ[env_flag] = "1"

    if force_reexec:
        # Re-execute the current process so ld.so picks up LD_LIBRARY_PATH
        try:
            os.execv(sys.executable, [sys.executable] + sys.argv)
        except Exception as e:
            print(f"[Warning] Failed to re-exec Python process with updated LD_LIBRARY_PATH: {e}")
            return False

    return True
