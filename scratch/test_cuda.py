import os
import sys

# Add MSVC to PATH
msvc_bin = r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64"
if msvc_bin not in os.environ["PATH"]:
    os.environ["PATH"] = msvc_bin + ";" + os.environ["PATH"]

import torch
from torch.utils.cpp_extension import load_inline
import time

cpp_source = "void run_dummy(torch::Tensor out);"

cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void dummy_kernel(float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = 42.0f;
    }
}

void run_dummy(torch::Tensor out) {
    int n = out.numel();
    dummy_kernel<<<(n + 255) / 256, 256>>>(out.data_ptr<float>(), n);
}
"""

print("Compiling inline CUDA extension...")
t0 = time.time()
mod = load_inline(
    name="dummy_cuda_fixed2",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["run_dummy"],
    with_cuda=True,
    extra_cuda_cflags=["-DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING", "-Xcompiler", "/Zc:preprocessor"],
    extra_cflags=["/Zc:preprocessor"],
    verbose=False
)
print(f"Compilation took {time.time() - t0:.2f}s")
t = torch.zeros(10, device="cuda")
mod.run_dummy(t)
torch.cuda.synchronize()
print("CUDA kernel output:", t)
