#include <cuda_runtime.h>

extern "C" __declspec(dllexport) void dummy_cuda_add(float* out, const float* in, int n, cudaStream_t stream) {
    // simple test
}
