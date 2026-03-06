#include "cudeep/kernels/norm.cuh"
#include "cudeep/ptx_intrinsics.cuh"
#include "cudeep/error.cuh"

namespace cudeep {
namespace kernels {

// ---------------------------------------------------------------------------
// Welford's online algorithm for single-pass mean + variance.
// ---------------------------------------------------------------------------

template <typename T>
struct WelfordState {
    T mean;
    T m2;
    int count;
};

template <typename T>
__device__ __forceinline__ WelfordState<T> welford_combine(
    WelfordState<T> a, WelfordState<T> b
) {
    if (b.count == 0) return a;
    if (a.count == 0) return b;
    int n = a.count + b.count;
    T delta = b.mean - a.mean;
    T new_mean = a.mean + delta * T(b.count) / T(n);
    T new_m2 = a.m2 + b.m2 + delta * delta * T(a.count) * T(b.count) / T(n);
    return {new_mean, new_m2, n};
}

template <typename T>
__device__ WelfordState<T> warp_welford_reduce(WelfordState<T> state) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        WelfordState<T> other;
        if constexpr (sizeof(T) == 4) {
            other.mean  = __shfl_down_sync(0xffffffff, state.mean, offset);
            other.m2    = __shfl_down_sync(0xffffffff, state.m2, offset);
            other.count = __shfl_down_sync(0xffffffff, state.count, offset);
        } else {
            other.mean  = shfl_down_double(state.mean, offset);
            other.m2    = shfl_down_double(state.m2, offset);
            other.count = __shfl_down_sync(0xffffffff, state.count, offset);
        }
        state = welford_combine(state, other);
    }
    return state;
}

template <typename T>
__device__ WelfordState<T> block_welford_reduce(WelfordState<T> state) {
    __shared__ T s_mean[32];
    __shared__ T s_m2[32];
    __shared__ int s_count[32];

    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    state = warp_welford_reduce(state);

    if (lane == 0) {
        s_mean[wid]  = state.mean;
        s_m2[wid]    = state.m2;
        s_count[wid] = state.count;
    }
    __syncthreads();

    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    if (wid == 0) {
        state = (threadIdx.x < num_warps)
            ? WelfordState<T>{s_mean[threadIdx.x], s_m2[threadIdx.x], s_count[threadIdx.x]}
            : WelfordState<T>{T(0), T(0), 0};
        state = warp_welford_reduce(state);
    }
    return state;
}

// ---------------------------------------------------------------------------
// BatchNorm2d — FUSED single-kernel: Welford stats + affine + running update
//
// One block per channel.  Two phases:
//   Phase 1: Welford online reduction over all (batch × spatial) elements
//   Phase 2: Re-read input, normalize, write output
//
// Eliminates the 3-kernel launch overhead and temporary batch_mean/batch_var
// allocations of the prior implementation.
// ---------------------------------------------------------------------------

template <typename T>
__global__ void batchnorm_fused_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* running_mean, T* running_var,
    int batch, int channels, int spatial,
    float eps, float momentum, bool update_running
) {
    int c = blockIdx.x;
    if (c >= channels) return;

    // Phase 1: Welford single-pass mean + variance with vectorized loads
    WelfordState<T> state = {T(0), T(0), 0};
    constexpr int VW = Vec4<T>::width;
    for (int b = 0; b < batch; ++b) {
        const T* in_ptr = input + (b * channels + c) * spatial;
        int vec_spatial = spatial / VW;
        for (int vi = threadIdx.x; vi < vec_spatial; vi += blockDim.x) {
            auto v = Vec4<T>::load(&in_ptr[vi * VW]);
            T vals[VW];
            if constexpr (sizeof(T) == 4) { vals[0]=v.x; vals[1]=v.y; vals[2]=v.z; vals[3]=v.w; }
            else                           { vals[0]=v.x; vals[1]=v.y; }
            #pragma unroll
            for (int j = 0; j < VW; ++j) {
                state.count++;
                T delta = vals[j] - state.mean;
                state.mean += delta / T(state.count);
                T delta2 = vals[j] - state.mean;
                state.m2 += delta * delta2;
            }
        }
        for (int si = vec_spatial * VW + threadIdx.x; si < spatial; si += blockDim.x) {
            T val = in_ptr[si];
            state.count++;
            T delta = val - state.mean;
            state.mean += delta / T(state.count);
            T delta2 = val - state.mean;
            state.m2 += delta * delta2;
        }
    }

    state = block_welford_reduce(state);

    __shared__ T s_mean, s_inv_std;
    if (threadIdx.x == 0) {
        T variance = (state.count > 0) ? state.m2 / T(state.count) : T(0);
        s_mean = state.mean;
        if constexpr (sizeof(T) == 4)
            s_inv_std = ptx::sfu_rsqrt(static_cast<float>(variance + T(eps)));
        else
            s_inv_std = T(1) / sqrt(variance + T(eps));

        if (update_running && running_mean && running_var) {
            running_mean[c] = T(1.0f - momentum) * running_mean[c] + T(momentum) * state.mean;
            running_var[c]  = T(1.0f - momentum) * running_var[c]  + T(momentum) * variance;
        }
    }
    __syncthreads();

    T mean    = s_mean;
    T inv_std = s_inv_std;
    T w       = weight[c];
    T b_val   = bias[c];

    // Phase 2: normalize + affine with vectorized stores
    for (int b_idx = 0; b_idx < batch; ++b_idx) {
        const T* in_ptr  = input  + (b_idx * channels + c) * spatial;
        T*       out_ptr = output + (b_idx * channels + c) * spatial;
        int vec_spatial = spatial / VW;
        for (int vi = threadIdx.x; vi < vec_spatial; vi += blockDim.x) {
            auto v = Vec4<T>::load(&in_ptr[vi * VW]);
            typename Vec4<T>::type vo;
            if constexpr (sizeof(T) == 4) {
                vo.x = w * (v.x - mean) * inv_std + b_val;
                vo.y = w * (v.y - mean) * inv_std + b_val;
                vo.z = w * (v.z - mean) * inv_std + b_val;
                vo.w = w * (v.w - mean) * inv_std + b_val;
            } else {
                vo.x = w * (v.x - mean) * inv_std + b_val;
                vo.y = w * (v.y - mean) * inv_std + b_val;
            }
            Vec4<T>::store(&out_ptr[vi * VW], vo);
        }
        for (int si = vec_spatial * VW + threadIdx.x; si < spatial; si += blockDim.x)
            out_ptr[si] = w * (in_ptr[si] - mean) * inv_std + b_val;
    }
}

// Inference path: just apply normalization using pre-computed running stats
template <typename T>
__global__ void batchnorm_apply_kernel(
    const T* input, T* output,
    const T* mean, const T* var,
    const T* weight, const T* bias,
    int total, int channels, int spatial, float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int c = (idx / spatial) % channels;
    T inv_std;
    if constexpr (sizeof(T) == 4)
        inv_std = ptx::sfu_rsqrt(static_cast<float>(var[c] + T(eps)));
    else
        inv_std = T(1) / sqrt(var[c] + T(eps));
    output[idx] = weight[c] * (input[idx] - mean[c]) * inv_std + bias[c];
}

template <typename T>
void launch_batchnorm_forward_kernel(
    const T* input, T* output,
    const T* weight, const T* bias,
    T* running_mean, T* running_var,
    int batch, int channels, int spatial,
    float eps, float momentum, bool training,
    cudaStream_t stream
) {
    if (training) {
        int threads = min(DEFAULT_BLOCK_SIZE, 1024);
        batchnorm_fused_kernel<<<channels, threads, 0, stream>>>(
            input, output, weight, bias,
            running_mean, running_var,
            batch, channels, spatial,
            eps, momentum, true);
        CUDEEP_CHECK_LAST_KERNEL();
    } else {
        int total = batch * channels * spatial;
        int threads = DEFAULT_BLOCK_SIZE;
        int blocks = ceil_div(total, threads);
        batchnorm_apply_kernel<<<blocks, threads, 0, stream>>>(
            input, output, running_mean, running_var, weight, bias,
            total, channels, spatial, eps);
        CUDEEP_CHECK_LAST_KERNEL();
    }
}

// ---------------------------------------------------------------------------
// LayerNorm — single-pass Welford + warp-shuffle reduction
// ---------------------------------------------------------------------------

template <typename T>
__global__ void layernorm_forward_kernel(
    const T* input, T* output,
    const T* weight, const T* bias,
    int batch_size, int normalized_size, float eps
) {
    int b = blockIdx.x;
    if (b >= batch_size) return;

    const T* in_row  = input  + b * normalized_size;
    T*       out_row = output + b * normalized_size;

    WelfordState<T> state = {T(0), T(0), 0};
    for (int i = threadIdx.x; i < normalized_size; i += blockDim.x) {
        T val = in_row[i];
        state.count++;
        T delta = val - state.mean;
        state.mean += delta / T(state.count);
        T delta2 = val - state.mean;
        state.m2 += delta * delta2;
    }

    state = block_welford_reduce(state);

    __shared__ T s_mean, s_inv_std;
    if (threadIdx.x == 0) {
        s_mean = state.mean;
        T variance = (state.count > 0) ? state.m2 / T(state.count) : T(0);
        if constexpr (sizeof(T) == 4)
            s_inv_std = ptx::sfu_rsqrt(static_cast<float>(variance + T(eps)));
        else
            s_inv_std = T(1) / sqrt(variance + T(eps));
    }
    __syncthreads();

    T mean    = s_mean;
    T inv_std = s_inv_std;

    constexpr int VW = Vec4<T>::width;
    int vec_ns = normalized_size / VW;
    for (int i = threadIdx.x; i < vec_ns; i += blockDim.x) {
        auto vi = Vec4<T>::load(&in_row[i * VW]);
        auto vw = Vec4<T>::load(&weight[i * VW]);
        auto vb = Vec4<T>::load(&bias[i * VW]);
        typename Vec4<T>::type vo;
        if constexpr (sizeof(T) == 4) {
            vo.x = vw.x * (vi.x - mean) * inv_std + vb.x;
            vo.y = vw.y * (vi.y - mean) * inv_std + vb.y;
            vo.z = vw.z * (vi.z - mean) * inv_std + vb.z;
            vo.w = vw.w * (vi.w - mean) * inv_std + vb.w;
        } else {
            vo.x = vw.x * (vi.x - mean) * inv_std + vb.x;
            vo.y = vw.y * (vi.y - mean) * inv_std + vb.y;
        }
        Vec4<T>::store(&out_row[i * VW], vo);
    }
    for (int i = vec_ns * VW + threadIdx.x; i < normalized_size; i += blockDim.x) {
        T x_hat = (in_row[i] - mean) * inv_std;
        out_row[i] = weight[i] * x_hat + bias[i];
    }
}

template <typename T>
void launch_layernorm_forward_kernel(
    const T* input, T* output,
    const T* weight, const T* bias,
    int batch_size, int normalized_size, float eps,
    cudaStream_t stream
) {
    int threads = DEFAULT_BLOCK_SIZE;
    layernorm_forward_kernel<<<batch_size, threads, 0, stream>>>(
        input, output, weight, bias, batch_size, normalized_size, eps);
    CUDEEP_CHECK_LAST_KERNEL();
}

// ---------------------------------------------------------------------------
// RMSNorm — used by LLaMA, Mistral, etc.
//
// RMSNorm(x) = x * rsqrt(mean(x²) + eps) * weight
//
// Simpler than LayerNorm: no mean subtraction, no bias.
// Single-pass: accumulate sum of squares with vectorized loads, then normalize.
// ---------------------------------------------------------------------------

template <typename T>
__global__ void rmsnorm_forward_kernel(
    const T* input, T* output,
    const T* weight,
    int batch_size, int normalized_size, float eps
) {
    int b = blockIdx.x;
    if (b >= batch_size) return;

    const T* in_row  = input  + b * normalized_size;
    T*       out_row = output + b * normalized_size;

    // Phase 1: compute sum of squares with vectorized loads
    T sum_sq = T(0);
    constexpr int VW = Vec4<T>::width;
    int vec_ns = normalized_size / VW;

    for (int i = threadIdx.x; i < vec_ns; i += blockDim.x) {
        auto v = Vec4<T>::load(&in_row[i * VW]);
        if constexpr (sizeof(T) == 4) {
            sum_sq += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
        } else {
            sum_sq += v.x*v.x + v.y*v.y;
        }
    }
    for (int i = vec_ns * VW + threadIdx.x; i < normalized_size; i += blockDim.x) {
        T val = in_row[i];
        sum_sq += val * val;
    }

    sum_sq = block_reduce_sum(sum_sq);

    __shared__ T s_rms_inv;
    if (threadIdx.x == 0) {
        T rms_sq = sum_sq / T(normalized_size) + T(eps);
        if constexpr (sizeof(T) == 4)
            s_rms_inv = ptx::sfu_rsqrt(static_cast<float>(rms_sq));
        else
            s_rms_inv = T(1) / sqrt(rms_sq);
    }
    __syncthreads();

    T rms_inv = s_rms_inv;

    // Phase 2: normalize with vectorized reads/writes
    for (int i = threadIdx.x; i < vec_ns; i += blockDim.x) {
        auto vi = Vec4<T>::load(&in_row[i * VW]);
        auto vw = Vec4<T>::load(&weight[i * VW]);
        typename Vec4<T>::type vo;
        if constexpr (sizeof(T) == 4) {
            vo.x = vi.x * rms_inv * vw.x;
            vo.y = vi.y * rms_inv * vw.y;
            vo.z = vi.z * rms_inv * vw.z;
            vo.w = vi.w * rms_inv * vw.w;
        } else {
            vo.x = vi.x * rms_inv * vw.x;
            vo.y = vi.y * rms_inv * vw.y;
        }
        Vec4<T>::store(&out_row[i * VW], vo);
    }
    for (int i = vec_ns * VW + threadIdx.x; i < normalized_size; i += blockDim.x)
        out_row[i] = in_row[i] * rms_inv * weight[i];
}

template <typename T>
void launch_rmsnorm_forward_kernel(
    const T* input, T* output,
    const T* weight,
    int batch_size, int normalized_size, float eps,
    cudaStream_t stream
) {
    int threads = DEFAULT_BLOCK_SIZE;
    rmsnorm_forward_kernel<<<batch_size, threads, 0, stream>>>(
        input, output, weight, batch_size, normalized_size, eps);
    CUDEEP_CHECK_LAST_KERNEL();
}

// ---- Explicit instantiations ----

template void launch_batchnorm_forward_kernel<float>(const float*, float*, const float*, const float*, float*, float*, int, int, int, float, float, bool, cudaStream_t);
template void launch_batchnorm_forward_kernel<double>(const double*, double*, const double*, const double*, double*, double*, int, int, int, float, float, bool, cudaStream_t);
template void launch_layernorm_forward_kernel<float>(const float*, float*, const float*, const float*, int, int, float, cudaStream_t);
template void launch_layernorm_forward_kernel<double>(const double*, double*, const double*, const double*, int, int, float, cudaStream_t);
template void launch_rmsnorm_forward_kernel<float>(const float*, float*, const float*, int, int, float, cudaStream_t);
template void launch_rmsnorm_forward_kernel<double>(const double*, double*, const double*, int, int, float, cudaStream_t);

}  // namespace kernels
}  // namespace cudeep
