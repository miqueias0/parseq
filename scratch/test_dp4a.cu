#include <cuda_runtime.h>
#include <stdio.h>

__global__ void test_dp4a_kernel(int* out) {
    // 4 bytes: 1, 2, 3, 4
    int a = 0x04030201;
    // 4 bytes: 1, 1, 1, 1
    int b = 0x01010101;
    out[0] = __dp4a(a, b, 0); // 1*1 + 2*1 + 3*1 + 4*1 = 10
}

extern "C" __declspec(dllexport) int run_test_dp4a() {
    int* d_out;
    int h_out = 0;
    cudaMalloc(&d_out, sizeof(int));
    test_dp4a_kernel<<<1, 1>>>(d_out);
    cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_out);
    return h_out;
}
