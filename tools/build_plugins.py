#!/usr/bin/env python3
"""Cross-platform CUDA plugin compiler for TensorRT quantization extensions:
- INT-FlashAttention (Algorithm 1, arXiv:2409.16997v2)
- SageAttention (arXiv:2410.02367v9)
- Integer LayerNorm (I-BERT Section 3.6 / Algorithm 4)
- Integer GELU (I-BERT Section 3.4)
- Integer Softmax (I-BERT Section 3.5)

Produces:
- Linux:   strhub/quant/plugins/parseq_plugins.so
- Windows: strhub/quant/plugins/parseq_plugins.dll
"""

import os
import sys
import shutil
import subprocess
from typing import List, Tuple, Optional


def get_nvcc_version(nvcc_path: str) -> float:
    """Extracts major.minor version from nvcc."""
    try:
        res = subprocess.run([nvcc_path, "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        import re
        match = re.search(r"release\s+(\d+\.\d+)", res.stdout)
        if match:
            return float(match.group(1))
    except Exception:
        pass
    return 0.0


def find_nvcc() -> Optional[str]:
    """Finds the newest available nvcc executable on the system."""
    candidates = []

    # 1. Check versioned CUDA directories on Linux (newest first)
    versioned_linux = [
        "/usr/local/cuda-12.6/bin/nvcc",
        "/usr/local/cuda-12.5/bin/nvcc",
        "/usr/local/cuda-12.4/bin/nvcc",
        "/usr/local/cuda-12.3/bin/nvcc",
        "/usr/local/cuda-12.2/bin/nvcc",
        "/usr/local/cuda-12.1/bin/nvcc",
        "/usr/local/cuda-12.0/bin/nvcc",
        "/usr/local/cuda-12/bin/nvcc",
        "/usr/local/cuda-11.8/bin/nvcc",
        "/usr/local/cuda-11/bin/nvcc",
        "/usr/local/cuda/bin/nvcc",
        "/usr/bin/nvcc",
        "/opt/cuda/bin/nvcc",
    ]
    for c in versioned_linux:
        if os.path.isfile(c) and c not in candidates:
            candidates.append(c)

    # 2. Check CUDA_HOME / CUDA_PATH
    for env_var in ["CUDA_HOME", "CUDA_PATH"]:
        if env_var in os.environ:
            bin_name = "nvcc.exe" if sys.platform.startswith("win") else "nvcc"
            c = os.path.join(os.environ[env_var], "bin", bin_name)
            if os.path.isfile(c) and c not in candidates:
                candidates.append(c)

    # 3. Check PATH
    which_nvcc = shutil.which("nvcc")
    if which_nvcc and os.path.isfile(which_nvcc) and which_nvcc not in candidates:
        candidates.append(which_nvcc)

    # 4. Check PyTorch installation directory
    try:
        from torch.utils.cpp_extension import CUDA_HOME
        if CUDA_HOME:
            bin_name = "nvcc.exe" if sys.platform.startswith("win") else "nvcc"
            c = os.path.join(CUDA_HOME, "bin", bin_name)
            if os.path.isfile(c) and c not in candidates:
                candidates.append(c)
    except Exception:
        pass

    if not candidates:
        return None

    # Sort candidates by release version descending to pick newest CUDA toolkit
    candidates.sort(key=get_nvcc_version, reverse=True)
    return candidates[0]


def find_ccbin() -> Optional[str]:
    """Finds a compatible host C++ compiler (e.g. g++-11, gcc-11) if available on Linux."""
    for cc in ["g++-11", "gcc-11", "g++-10", "gcc-10"]:
        p = shutil.which(cc)
        if p and os.path.isfile(p):
            return p
        for loc in [f"/usr/bin/{cc}", f"/usr/local/bin/{cc}"]:
            if os.path.isfile(loc):
                return loc
    return None


def get_host_compiler_flags(specified_ccbin: Optional[str] = None) -> List[str]:
    """Returns nvcc flags to resolve GCC version checks on Linux."""
    if sys.platform.startswith("win"):
        return []
    flags = []
    ccbin = specified_ccbin or find_ccbin()
    if ccbin:
        flags.extend(["-ccbin", ccbin])
    flags.append("-allow-unsupported-compiler")
    return flags


def filter_supported_architectures(nvcc_bin: str, arch_candidates: List[str], host_flags: Optional[List[str]] = None) -> List[str]:
    """Tests each architecture candidate against nvcc using a minimal temporary source."""
    import tempfile
    supported = []
    with tempfile.NamedTemporaryFile(suffix=".cu", mode="w", delete=False) as f:
        f.write("__global__ void _test_arch_kernel() {}\n")
        temp_cu = f.name
    try:
        temp_out = temp_cu + (".obj" if sys.platform.startswith("win") else ".o")
        for arch in arch_candidates:
            cmd = [nvcc_bin]
            if host_flags:
                cmd.extend(host_flags)
            cmd.extend(["-gencode", f"arch={arch}", "-c", temp_cu, "-o", temp_out])
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode == 0:
                supported.append(arch)
            if os.path.exists(temp_out):
                try:
                    os.remove(temp_out)
                except OSError:
                    pass
    finally:
        if os.path.exists(temp_cu):
            try:
                os.remove(temp_cu)
            except OSError:
                pass
    return supported


def get_target_architectures(nvcc_bin: Optional[str] = None, specified_arch: Optional[str] = None, host_flags: Optional[List[str]] = None) -> List[str]:
    """Detects available GPU architectures and validates them against nvcc."""
    if specified_arch:
        arch_clean = specified_arch.strip()
        if not arch_clean.startswith("compute_"):
            num = arch_clean.replace("sm_", "").replace("sm", "")
            candidates = [f"compute_{num},code=sm_{num}"]
        else:
            candidates = [arch_clean]
        if nvcc_bin:
            filtered = filter_supported_architectures(nvcc_bin, candidates, host_flags=host_flags)
            if filtered:
                return filtered
        return candidates

    archs = set()

    # 1. Check current torch cuda device (e.g. RTX 2080 Ti -> compute_75,code=sm_75)
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            for i in range(torch.cuda.device_count()):
                cap = torch.cuda.get_device_capability(i)
                archs.add(f"compute_{cap[0]}{cap[1]},code=sm_{cap[0]}{cap[1]}")
    except Exception:
        pass

    # If physical GPUs are detected, use their exact architecture(s)
    if archs:
        candidate_list = sorted(list(archs))
        if nvcc_bin:
            filtered = filter_supported_architectures(nvcc_bin, candidate_list, host_flags=host_flags)
            if filtered:
                return filtered
        return candidate_list

    # 2. Fallback common architectures when no GPU is detected (e.g. build server)
    fallback_candidates = [
        "compute_75,code=sm_75",  # Turing (RTX 2080 Ti, T4)
        "compute_86,code=sm_86",  # Ampere (RTX 3080/3090, A40)
        "compute_80,code=sm_80",  # Ampere (A100)
        "compute_89,code=sm_89",  # Ada Lovelace (RTX 4090, L4)
    ]

    if nvcc_bin:
        supported = filter_supported_architectures(nvcc_bin, fallback_candidates, host_flags=host_flags)
        if supported:
            return supported

    return ["compute_75,code=sm_75"]


def compile_cuda_plugins(
    output_path: Optional[str] = None,
    arch: Optional[str] = None,
    ccbin: Optional[str] = None,
    verbose: bool = True
) -> Tuple[bool, str]:
    """Compiles the custom CUDA kernels into a shared library (.so on Linux, .dll on Windows)."""
    plugin_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "strhub", "quant", "plugins")
    
    is_windows = sys.platform.startswith("win")
    lib_name = "parseq_plugins.dll" if is_windows else "parseq_plugins.so"
    
    if output_path is None:
        output_path = os.path.join(plugin_dir, lib_name)
    else:
        output_path = os.path.abspath(output_path)

    nvcc = find_nvcc()
    if not nvcc:
        err_msg = (
            "nvcc compiler not found in PATH or standard CUDA directories.\n"
            "Ensure CUDA Toolkit is installed and accessible.\n"
            "Standard locations checked: /usr/local/cuda-12.4, /usr/local/cuda, etc."
        )
        if verbose:
            print(f"[build_plugins] ERROR: {err_msg}", file=sys.stderr)
        return False, err_msg

    sources = [
        os.path.join(plugin_dir, "int_flashattention_kernel.cu"),
        os.path.join(plugin_dir, "sage_attention_kernel.cu"),
        os.path.join(plugin_dir, "integer_nonlinear_kernels.cu"),
    ]

    for src in sources:
        if not os.path.isfile(src):
            err_msg = f"Source file not found: {src}"
            if verbose:
                print(f"[build_plugins] ERROR: {err_msg}", file=sys.stderr)
            return False, err_msg

    host_flags = get_host_compiler_flags(specified_ccbin=ccbin)
    target_archs = get_target_architectures(nvcc_bin=nvcc, specified_arch=arch, host_flags=host_flags)
    arch_flags = []
    for a in target_archs:
        arch_flags.extend(["-gencode", f"arch={a}"])

    cmd = [
        nvcc,
        "-O3",
        "-shared",
    ]
    cmd.extend(host_flags)

    if is_windows:
        cmd.extend(["-Xcompiler", "/MD"])
    else:
        cmd.extend(["-Xcompiler", "-fPIC"])

    cmd.extend(arch_flags)
    cmd.extend(sources)
    cmd.extend(["-o", output_path])

    if verbose:
        nvcc_ver = get_nvcc_version(nvcc)
        print(f"[build_plugins] Compiling CUDA plugins using: {nvcc} (CUDA release {nvcc_ver})")
        print(f"[build_plugins] Target architectures: {', '.join(target_archs)}")
        if host_flags:
            print(f"[build_plugins] Host compiler flags: {' '.join(host_flags)}")
        print(f"[build_plugins] Output: {output_path}")
        print(f"[build_plugins] Command: {' '.join(cmd)}")

    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        if verbose:
            if res.stdout.strip():
                print(res.stdout)
            print(f"[build_plugins] ✓ Successfully compiled {output_path} ({os.path.getsize(output_path) / 1024:.1f} KB)")
        return True, output_path
    except subprocess.CalledProcessError as e:
        err_msg = f"nvcc compilation failed with exit code {e.returncode}:\n{e.stderr}\n{e.stdout}"
        if verbose:
            print(f"[build_plugins] ERROR: {err_msg}", file=sys.stderr)
        return False, err_msg


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compile PARSeq CUDA plugins for TensorRT.")
    parser.add_argument("--output", type=str, default=None, help="Output library path (.so / .dll)")
    parser.add_argument("--arch", type=str, default=None, help="Specific GPU architecture (e.g. sm_75, 75, compute_75,code=sm_75)")
    parser.add_argument("--ccbin", type=str, default=None, help="Host compiler binary path (e.g. g++-11, /usr/bin/gcc-11)")
    args = parser.parse_args()

    success, result = compile_cuda_plugins(output_path=args.output, arch=args.arch, ccbin=args.ccbin, verbose=True)
    if not success:
        sys.exit(1)
