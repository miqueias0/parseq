#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>

#define WARP_SIZE 32

// Warp reduction for float maximum
__device__ __forceinline__ float warp_reduce_max_sage(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Warp reduction for int sum
__device__ __forceinline__ int warp_reduce_sum_int_sage(int val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Fused SageAttention Kernel (Algorithm 1 from arXiv:2410.02367v9)
// Features:
// 1. Token-averaged Key smoothing: K_smooth = K - mean(K, tokens)
//    Guarantees softmax invariance while suppressing channel-wise key outliers.
// 2. INT8 x INT8 -> INT32 GEMM for Q @ K_smooth^T
// 3. Numerically stable online Softmax
// 4. SAGEAttn-B (mode=0, FP16/FP32 V accumulation) or SAGEAttn-vB (mode=1, fully INT8 GEMM-2)
//
// Grid: (N, H, B) -> blockIdx.x = query_token_i, blockIdx.y = head_h, blockIdx.z = batch_b
// Block: (D, 1, 1) -> threadIdx.x = feature_d (D = 32 or 64)
extern "C" __global__ void sage_attention_forward_kernel(
    float* __restrict__ out,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    int B,
    int H,
    int N,
    int S,
    int D,
    float scale,
    int mode // 0 = SAGEAttn-B (FP16/FP32 V), 1 = SAGEAttn-vB (fully INT8 V)
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

    // 1. Preprocessing: Compute token-averaged Key mean_k[d] for this head dimension (Eq. 6)
    // Each thread d computes the mean across all key tokens S for its feature channel d
    float sum_k_d = 0.0f;
    float max_v_d = 0.0f;
    for (int s = 0; s < S; ++s) {
        size_t offset_s = kv_base + s * kv_stride_s + d;
        float kval = k[offset_s];
        sum_k_d += kval;
        if (mode == 1) {
            max_v_d = fmaxf(max_v_d, fabsf(v[offset_s]));
        }
    }
    float mean_k_d = sum_k_d / (float)S;
    float S_v_d = fmaxf(max_v_d / 127.0f, 1e-8f); // Per-channel scale for V (SAGEAttn-vB)

    // 2. Load and quantize Query component for this thread (scaled by 1/sqrt(d))
    float q_val = q[q_offset + d] * scale;
    float q_abs = fabsf(q_val);

    __shared__ float s_warp_flt[2];
    __shared__ int s_warp_int[2];
    __shared__ float s_q_max;

    float warp_q_max = warp_reduce_max_sage(q_abs);
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
    int q_int = (int)roundf(q_val / S_q);
    q_int = max(-127, min(127, q_int));

    // Online softmax and output accumulator initialization
    float m_i = -1e20f;
    float l_i = 0.0f;
    float acc_d = 0.0f;

    // 3. Loop over Key/Value tokens j in [0, S-1] (Algorithm 1)
    for (int j = 0; j < S; ++j) {
        size_t kv_offset = kv_base + j * kv_stride_s + d;
        float raw_k_val = k[kv_offset];
        float v_val = v[kv_offset];

        // Apply Key Smoothing: k_smooth = raw_k - mean_k
        float k_smooth_val = raw_k_val - mean_k_d;

        // Find max(|k_smooth|) across dimension D for token j
        float k_abs = fabsf(k_smooth_val);
        float warp_k_max = warp_reduce_max_sage(k_abs);
        if (lane == 0) s_warp_flt[warp_id] = warp_k_max;
        __syncthreads();

        __shared__ float s_k_max;
        if (d == 0) {
            float r_k = s_warp_flt[0];
            if (num_warps > 1) r_k = fmaxf(r_k, s_warp_flt[1]);
            s_k_max = r_k;
        }
        __syncthreads();

        float S_k = fmaxf(s_k_max / 127.0f, 1e-8f);

        // Quantize k_smooth to INT8
        int k_int = (int)roundf(k_smooth_val / S_k);
        k_int = max(-127, min(127, k_int));

        // GEMM-1: q_int * k_int -> INT32
        int prod = q_int * k_int;
        int warp_sum = warp_reduce_sum_int_sage(prod);
        if (lane == 0) s_warp_int[warp_id] = warp_sum;
        __syncthreads();

        __shared__ int s_S_int;
        if (d == 0) {
            int total_p = s_warp_int[0];
            if (num_warps > 1) total_p += s_warp_int[1];
            s_S_int = total_p;
        }
        __syncthreads();

        // Rescale INT32 dot product: S_ij = S_int * (S_q * S_k)
        float S_ij = (float)s_S_int * (S_q * S_k);

        // Online Softmax update (Algorithm 1 Line 10)
        float m_prev = m_i;
        m_i = fmaxf(m_prev, S_ij);
        float alpha = expf(m_prev - m_i);
        float p_ij = expf(S_ij - m_i);
        l_i = alpha * l_i + p_ij;

        // GEMM-2: Output accumulation
        if (mode == 1) {
            // SAGEAttn-vB: Fully INT8 second GEMM
            const float S_p = 1.0f / 127.0f;
            int p_int = (int)roundf(p_ij / S_p);
            p_int = max(0, min(127, p_int));

            // Quantize v per-channel:
            int v_int = (int)roundf(v_val / S_v_d);
            v_int = max(-127, min(127, v_int));

            // INT8 GEMM output: (p_int * v_int) * (S_p * S_v_d)
            float v_contrib = ((float)(p_int * v_int)) * (S_p * S_v_d);
            acc_d = alpha * acc_d + v_contrib;
        } else {
            // SAGEAttn-B: Standard high-precision V accumulation
            acc_d = alpha * acc_d + p_ij * v_val;
        }
    }

    // 4. Normalization by sum l_i (Algorithm 1 Line 12)
    float final_out = acc_d / fmaxf(l_i, 1e-8f);
    out[q_offset + d] = final_out;
}

#if defined(_WIN32) || defined(__CYGWIN__)
  #define PARSEQ_PLUGIN_EXPORT __declspec(dllexport)
#else
  #define PARSEQ_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

// C-accessible entry point for ctypes / C++ / TensorRT plugin
extern "C" PARSEQ_PLUGIN_EXPORT void run_sage_attention(
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
    int mode,
    cudaStream_t stream
) {
    dim3 grid(N, H, B);
    dim3 block(D, 1, 1);
    sage_attention_forward_kernel<<<grid, block, 0, stream>>>(
        out, q, k, v, B, H, N, S, D, scale, mode
    );
}
