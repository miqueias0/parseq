#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>

#define WARP_SIZE 32
#define MAX_WARPS_PER_BLOCK 8

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max_flt(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// 1. Integer LayerNorm Kernel (I-BERT Section 3.6 & Algorithm 4)
// Grid: (num_rows)
// Block: (threads_per_row) e.g. 128 or 256
extern "C" __global__ void integer_layernorm_forward_kernel(
    float* __restrict__ out,
    const float* __restrict__ in,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int num_rows,
    int D,
    float eps
) {
    int row = blockIdx.x;
    if (row >= num_rows) return;

    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    const float* row_in = in + row * D;
    float* row_out = out + row * D;

    __shared__ float s_warp_vals[MAX_WARPS_PER_BLOCK];
    __shared__ float s_mean;
    __shared__ float s_var;

    // Phase 1: Compute Mean
    float sum = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        sum += row_in[d];
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0) {
        s_warp_vals[warp_id] = sum;
    }
    __syncthreads();

    if (tid == 0) {
        float total_sum = 0.0f;
        for (int w = 0; w < num_warps; ++w) {
            total_sum += s_warp_vals[w];
        }
        s_mean = total_sum / (float)D;
    }
    __syncthreads();
    float mean = s_mean;

    // Phase 2: Compute Variance
    float var_sum = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        float diff = row_in[d] - mean;
        var_sum += diff * diff;
    }
    var_sum = warp_reduce_sum(var_sum);
    if (lane == 0) {
        s_warp_vals[warp_id] = var_sum;
    }
    __syncthreads();

    if (tid == 0) {
        float total_var = 0.0f;
        for (int w = 0; w < num_warps; ++w) {
            total_var += s_warp_vals[w];
        }
        s_var = total_var / (float)D;
    }
    __syncthreads();
    float variance = s_var;

    // Phase 3: Integer Newton-Raphson Square Root (Algorithm 4)
    float n_val = fmaxf(variance + eps, 1e-8f);
    float xi = fmaxf(sqrtf(n_val), 1e-4f);
    #pragma unroll
    for (int it = 0; it < 4; ++it) {
        xi = 0.5f * (xi + n_val / xi);
    }
    float std = xi;

    // Phase 4: Normalize, scale, and bias
    for (int d = tid; d < D; d += blockDim.x) {
        float normed = (row_in[d] - mean) / std;
        float w = weight ? weight[d] : 1.0f;
        float b = bias ? bias[d] : 0.0f;
        row_out[d] = normed * w + b;
    }
}

// 2. Integer GELU Kernel (I-BERT Section 3.4)
// Element-wise polynomial approximation
extern "C" __global__ void integer_gelu_forward_kernel(
    float* __restrict__ out,
    const float* __restrict__ in,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    const float inv_sqrt2 = 0.7071067811865475f;
    const float a = -0.2888f;
    const float b = -1.769f;

    float x = in[idx];
    float x_scaled = x * inv_sqrt2;
    float sign = (x_scaled > 0.0f) ? 1.0f : ((x_scaled < 0.0f) ? -1.0f : 0.0f);
    float abs_x = fabsf(x_scaled);
    float clipped = fminf(abs_x, -b);
    float diff = clipped + b;
    float poly = a * (diff * diff) + 1.0f;
    float erf_approx = sign * poly;

    out[idx] = 0.5f * x * (1.0f + erf_approx);
}

// 3. Integer Softmax Kernel (I-BERT Section 3.5)
// Grid: (num_rows)
// Block: (threads_per_row) e.g. 128 or 256
extern "C" __global__ void integer_softmax_forward_kernel(
    float* __restrict__ out,
    const float* __restrict__ in,
    int num_rows,
    int N
) {
    int row = blockIdx.x;
    if (row >= num_rows) return;

    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    const float* row_in = in + row * N;
    float* row_out = out + row * N;

    __shared__ float s_warp_vals[MAX_WARPS_PER_BLOCK];
    __shared__ float s_row_max;
    __shared__ float s_sum_exp;

    // Phase 1: Row Maximum
    float max_val = -1e20f;
    for (int i = tid; i < N; i += blockDim.x) {
        max_val = fmaxf(max_val, row_in[i]);
    }
    max_val = warp_reduce_max_flt(max_val);
    if (lane == 0) {
        s_warp_vals[warp_id] = max_val;
    }
    __syncthreads();

    if (tid == 0) {
        float r_max = -1e20f;
        for (int w = 0; w < num_warps; ++w) {
            r_max = fmaxf(r_max, s_warp_vals[w]);
        }
        s_row_max = r_max;
    }
    __syncthreads();
    float row_max = s_row_max;

    // Phase 2: Compute exp approximation and sum
    const float ln2 = 0.6931471805599453f;
    const float a = 0.3585f;
    const float b_val = 1.353f;
    const float c_val = 0.344f;

    float sum_exp = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float x_shifted = row_in[i] - row_max; // <= 0
        float z = floorf(-x_shifted / ln2);
        float p = x_shifted + z * ln2;
        float p_b = p + b_val;
        float poly = a * (p_b * p_b) + c_val;
        float exp_val = poly * exp2f(-z);
        row_out[i] = exp_val; // temporarily store unnormalized
        sum_exp += exp_val;
    }
    sum_exp = warp_reduce_sum(sum_exp);
    if (lane == 0) {
        s_warp_vals[warp_id] = sum_exp;
    }
    __syncthreads();

    if (tid == 0) {
        float total_sum = 0.0f;
        for (int w = 0; w < num_warps; ++w) {
            total_sum += s_warp_vals[w];
        }
        s_sum_exp = fmaxf(total_sum, 1e-8f);
    }
    __syncthreads();
    float normalizer = s_sum_exp;

    // Phase 3: Normalize
    for (int i = tid; i < N; i += blockDim.x) {
        row_out[i] = row_out[i] / normalizer;
    }
}

#if defined(_WIN32) || defined(__CYGWIN__)
  #define PARSEQ_PLUGIN_EXPORT __declspec(dllexport)
#else
  #define PARSEQ_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

// Exported C functions
extern "C" PARSEQ_PLUGIN_EXPORT void run_integer_layernorm(
    float* out, const float* in, const float* weight, const float* bias,
    int num_rows, int D, float eps, cudaStream_t stream
) {
    int block_size = 256;
    integer_layernorm_forward_kernel<<<num_rows, block_size, 0, stream>>>(
        out, in, weight, bias, num_rows, D, eps
    );
}

extern "C" PARSEQ_PLUGIN_EXPORT void run_integer_gelu(
    float* out, const float* in, int total_elements, cudaStream_t stream
) {
    int block_size = 256;
    int grid = (total_elements + block_size - 1) / block_size;
    integer_gelu_forward_kernel<<<grid, block_size, 0, stream>>>(
        out, in, total_elements
    );
}

extern "C" PARSEQ_PLUGIN_EXPORT void run_integer_softmax(
    float* out, const float* in, int num_rows, int N, cudaStream_t stream
) {
    int block_size = 256;
    integer_softmax_forward_kernel<<<num_rows, block_size, 0, stream>>>(
        out, in, num_rows, N
    );
}
