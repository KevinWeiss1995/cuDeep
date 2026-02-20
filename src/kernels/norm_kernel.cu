#include "cudeep/kernels/norm.cuh"
#include "cudeep/error.cuh"

namespace cudeep {
namespace kernels {

// ---------------------------------------------------------------------------
// Welford's online algorithm for single-pass mean + variance.
// Each thread accumulates (count, mean, M2) and then we combine across the
// block using warp shuffles.
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
// BatchNorm2d — single-pass Welford stats + affine
// ---------------------------------------------------------------------------

template <typename T>
__global__ void batchnorm_stats_kernel(
    const T* input, T* mean, T* var,
    int batch, int channels, int spatial
) {
    int c = blockIdx.x;
    if (c >= channels) return;

    int total = batch * spatial;

    WelfordState<T> state = {T(0), T(0), 0};
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        int b = i / spatial;
        int s = i % spatial;
        T val = input[(b * channels + c) * spatial + s];
        state.count++;
        T delta = val - state.mean;
        state.mean += delta / T(state.count);
        T delta2 = val - state.mean;
        state.m2 += delta * delta2;
    }

    state = block_welford_reduce(state);

    if (threadIdx.x == 0) {
        mean[c] = state.mean;
        var[c]  = (state.count > 0) ? state.m2 / T(state.count) : T(0);
    }
}

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
    T inv_std = T(1) / sqrt(var[c] + T(eps));
    output[idx] = weight[c] * (input[idx] - mean[c]) * inv_std + bias[c];
}

template <typename T>
__global__ void batchnorm_update_running_kernel(
    T* running_mean, T* running_var,
    const T* batch_mean, const T* batch_var,
    int channels, float momentum
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;

    running_mean[c] = T(1.0f - momentum) * running_mean[c] + T(momentum) * batch_mean[c];
    running_var[c]  = T(1.0f - momentum) * running_var[c]  + T(momentum) * batch_var[c];
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
    int total = batch * channels * spatial;
    int threads = DEFAULT_BLOCK_SIZE;

    if (training) {
        T* batch_mean = nullptr;
        T* batch_var  = nullptr;
        CUDEEP_CHECK_CUDA(cudaMallocAsync(&batch_mean, channels * sizeof(T), stream));
        CUDEEP_CHECK_CUDA(cudaMallocAsync(&batch_var,  channels * sizeof(T), stream));

        batchnorm_stats_kernel<<<channels, threads, 0, stream>>>(
            input, batch_mean, batch_var, batch, channels, spatial);
        CUDEEP_CHECK_LAST_KERNEL();

        int blocks = ceil_div(total, threads);
        batchnorm_apply_kernel<<<blocks, threads, 0, stream>>>(
            input, output, batch_mean, batch_var, weight, bias,
            total, channels, spatial, eps);
        CUDEEP_CHECK_LAST_KERNEL();

        if (running_mean && running_var) {
            int rblocks = ceil_div(channels, threads);
            batchnorm_update_running_kernel<<<rblocks, threads, 0, stream>>>(
                running_mean, running_var, batch_mean, batch_var,
                channels, momentum);
            CUDEEP_CHECK_LAST_KERNEL();
        }

        CUDEEP_CHECK_CUDA(cudaFreeAsync(batch_mean, stream));
        CUDEEP_CHECK_CUDA(cudaFreeAsync(batch_var,  stream));
    } else {
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
        s_inv_std = T(1) / sqrt(variance + T(eps));
    }
    __syncthreads();

    T mean    = s_mean;
    T inv_std = s_inv_std;

    for (int i = threadIdx.x; i < normalized_size; i += blockDim.x) {
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

// ---- Explicit instantiations ----

template void launch_batchnorm_forward_kernel<float>(const float*, float*, const float*, const float*, float*, float*, int, int, int, float, float, bool, cudaStream_t);
template void launch_batchnorm_forward_kernel<double>(const double*, double*, const double*, const double*, double*, double*, int, int, int, float, float, bool, cudaStream_t);
template void launch_layernorm_forward_kernel<float>(const float*, float*, const float*, const float*, int, int, float, cudaStream_t);
template void launch_layernorm_forward_kernel<double>(const double*, double*, const double*, const double*, int, int, float, cudaStream_t);

}  // namespace kernels
}  // namespace cudeep
