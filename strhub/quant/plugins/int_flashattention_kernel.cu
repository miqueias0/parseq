#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>

#define WARP_SIZE 32

// Warp reduction for float maximum
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Warp reduction for int sum
__device__ __forceinline__ int warp_reduce_sum_int(int val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Fused INT-FlashAttention Kernel (Algorithm 1 from arXiv:2409.16997v2)
// Grid: (N, H, B) -> blockIdx.x = query_token_i, blockIdx.y = head_h, blockIdx.z = batch_b
// Block: (D, 1, 1) -> threadIdx.x = feature_d (D = 32 or 64)
extern "C" __global__ void int_flash_attention_forward_kernel(
    float* __restrict__ out,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    int B,
    int H,
    int N,
    int S,
    int D,
    float scale
) {
    int i = blockIdx.x; // Query token index [0, N-1]
    int h = blockIdx.y; // Head index [0, H-1]
    int b = blockIdx.z; // Batch index [0, B-1]
    int d = threadIdx.x; // Head dimension index [0, D-1]

    if (i >= N || h >= H || b >= B || d >= D) return;

    int lane = d % WARP_SIZE;
    int warp_id = d / WARP_SIZE;
    int num_warps = (D + WARP_SIZE - 1) / WARP_SIZE;

    // Strides for [B, H, N, D]
    size_t q_stride_b = (size_t)H * N * D;
    size_t q_stride_h = (size_t)N * D;
    size_t q_stride_n = (size_t)D;

    size_t kv_stride_b = (size_t)H * S * D;
    size_t kv_stride_h = (size_t)S * D;
    size_t kv_stride_s = (size_t)D;

    size_t q_offset = b * q_stride_b + h * q_stride_h + i * q_stride_n;
    size_t kv_base  = b * kv_stride_b + h * kv_stride_h;

    // 1. Load Query component for this thread
    float q_val = q[q_offset + d];

    // Find max(|q|) across head dimension D
    float q_abs = fabsf(q_val);
    __shared__ float s_warp_flt[2];
    __shared__ int s_warp_int[2];
    __shared__ float s_q_max;

    float warp_q_max = warp_reduce_max(q_abs);
    if (lane == 0) {
        s_warp_flt[warp_id] = warp_q_max;
    }
    __syncthreads();

    if (d == 0) {
        float r_max = s_warp_flt[0];
        if (num_warps > 1) r_max = fmaxf(r_max, s_warp_flt[1]);
        s_q_max = r_max;
    }
    __syncthreads();

    float S_q = fmaxf(s_q_max / 127.0f, 1e-8f);
    // Quantize q to INT8
    int q_int = (int)roundf(q_val / S_q);
    q_int = max(-127, min(127, q_int));

    // Initialize online softmax statistics and output accumulator
    float m_i = -1e20f;
    float l_i = 0.0f;
    float acc_d = 0.0f;

    // 2. Loop over key/value sequence tokens j in [0, S-1]
    for (int j = 0; j < S; ++j) {
        size_t kv_offset = kv_base + j * kv_stride_s + d;
        float k_val = k[kv_offset];
        float v_val = v[kv_offset];

        // Find max(|k|)
        float k_abs = fabsf(k_val);
        float warp_k_max = warp_reduce_max(k_abs);
        if (lane == 0) s_warp_flt[warp_id] = warp_k_max;
        __syncthreads();

        __shared__ float s_k_max;
        if (d == 0) {
            float r_k = s_warp_flt[0];
            if (num_warps > 1) r_k = fmaxf(r_k, s_warp_flt[1]);
            s_k_max = r_k;
        }
        __syncthreads();

        // Find max(|v|)
        float v_abs = fabsf(v_val);
        float warp_v_max = warp_reduce_max(v_abs);
        if (lane == 0) s_warp_flt[warp_id] = warp_v_max;
        __syncthreads();

        __shared__ float s_v_max;
        if (d == 0) {
            float r_v = s_warp_flt[0];
            if (num_warps > 1) r_v = fmaxf(r_v, s_warp_flt[1]);
            s_v_max = r_v;
        }
        __syncthreads();

        float S_k = fmaxf(s_k_max / 127.0f, 1e-8f);
        float S_v = fmaxf(s_v_max / 127.0f, 1e-8f);

        // Quantize k to INT8
        int k_int = (int)roundf(k_val / S_k);
        k_int = max(-127, min(127, k_int));

        // INT8 dot-product term: q_int * k_int -> INT32
        int prod = q_int * k_int;

        // Sum reduction across dimension D to get S_int(j)
        int warp_sum = warp_reduce_sum_int(prod);
        if (lane == 0) s_warp_int[warp_id] = warp_sum;
        __syncthreads();

        __shared__ int s_S_int;
        if (d == 0) {
            int total_p = s_warp_int[0];
            if (num_warps > 1) total_p += s_warp_int[1];
            s_S_int = total_p;
        }
        __syncthreads();

        // Rescale INT32 dot product: S_ij = S_int * (S_q * S_k * scale)
        float S_ij = (float)s_S_int * (S_q * S_k * scale);

        // Online Softmax update (Algorithm 1)
        float m_prev = m_i;
        m_i = fmaxf(m_prev, S_ij);
        float alpha = expf(m_prev - m_i);
        float p_ij = expf(S_ij - m_i);
        l_i = alpha * l_i + p_ij;

        // Quantize P_ij to INT8: P_int in [0, 127] with scale S_p = 1/127
        const float S_p = 1.0f / 127.0f;
        int p_int = (int)roundf(p_ij / S_p);
        p_int = max(0, min(127, p_int));

        // Quantize v to INT8
        int v_int = (int)roundf(v_val / S_v);
        v_int = max(-127, min(127, v_int));

        // INT8 GEMM for output: p_int * v_int (INT32) * (S_p * S_v)
        float v_contrib = ((float)(p_int * v_int)) * (S_p * S_v);

        // Update running output accumulator
        acc_d = alpha * acc_d + v_contrib;
    }

    // 3. Final normalization by l_i and write to output
    float final_out = acc_d / fmaxf(l_i, 1e-8f);
    out[q_offset + d] = final_out;
}

// C-accessible entry point for ctypes / C++
extern "C" __declspec(dllexport) void run_int_flash_attention(
    float* out,
    const float* q,
    const float* k,
    const float* v,
    int B,
    int H,
    int N,
    int S,
    int D,
    float scale,
    cudaStream_t stream
) {
    dim3 grid(N, H, B);
    dim3 block(D, 1, 1);
    int_flash_attention_forward_kernel<<<grid, block, 0, stream>>>(
        out, q, k, v, B, H, N, S, D, scale
    );
}
