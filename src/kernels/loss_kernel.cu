#include "cudeep/kernels/loss.cuh"
#include "cudeep/ptx_intrinsics.cuh"
#include "cudeep/error.cuh"

namespace {

template <typename T>
__device__ __forceinline__ T fast_exp_dispatch(T x) {
    if constexpr (sizeof(T) == 4)
        return cudeep::ptx::fast_exp_ptx(static_cast<float>(x));
    else
        return exp(x);
}

template <typename T>
__device__ __forceinline__ T fast_log_dispatch(T x) {
    if constexpr (sizeof(T) == 4)
        return cudeep::ptx::fast_log_ptx(static_cast<float>(x));
    else
        return log(x);
}

template <typename T>
__device__ __forceinline__ T fast_rcp_dispatch(T x) {
    if constexpr (sizeof(T) == 4)
        return cudeep::ptx::sfu_rcp(static_cast<float>(x));
    else
        return T(1) / x;
}

}  // anonymous namespace

namespace cudeep {
namespace kernels {

// ===========================================================================
// MSE Loss — grid-stride + warp-shuffle reduction (unchanged, already fast)
// ===========================================================================

template <typename T>
__global__ void mse_loss_kernel(const T* __restrict__ pred,
                                 const T* __restrict__ target,
                                 T* loss, int64_t n) {
    T acc = T(0);
    constexpr int VW = Vec4<T>::width;
    int64_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    int64_t vec_n  = n / VW;

    for (int64_t i = tid; i < vec_n; i += stride) {
        auto vp = Vec4<T>::load(&pred[i * VW]);
        auto vt = Vec4<T>::load(&target[i * VW]);
        if constexpr (sizeof(T) == 4) {
            float d0 = vp.x - vt.x, d1 = vp.y - vt.y;
            float d2 = vp.z - vt.z, d3 = vp.w - vt.w;
            acc += d0*d0 + d1*d1 + d2*d2 + d3*d3;
        } else {
            double d0 = vp.x - vt.x, d1 = vp.y - vt.y;
            acc += d0*d0 + d1*d1;
        }
    }
    for (int64_t i = vec_n * VW + tid; i < n; i += stride) {
        T diff = pred[i] - target[i];
        acc += diff * diff;
    }

    acc = block_reduce_sum(acc);
    if (threadIdx.x == 0)
        atomicAdd(loss, acc / static_cast<T>(n));
}

template <typename T>
void launch_mse_loss_kernel(const T* pred, const T* target, T* loss, int64_t n, cudaStream_t stream) {
    CUDEEP_CHECK_CUDA(cudaMemsetAsync(loss, 0, sizeof(T), stream));
    int threads = DEFAULT_BLOCK_SIZE;
    int blocks = min(ceil_div(static_cast<int>(n), threads), 256);
    mse_loss_kernel<<<blocks, threads, 0, stream>>>(pred, target, loss, n);
    CUDEEP_CHECK_LAST_KERNEL();
}

// ===========================================================================
// Softmax — online safe softmax with two passes fused where possible
//
// Pass 1: find row max + compute sum of exp(x - max) in a single reduction
//         (online normalization trick — we update max and running sum
//          simultaneously, avoiding a separate max pass).
// Pass 2: write exp(x - max) / sum
//
// This is the "online softmax" algorithm from Milakov & Gimelshein (2018).
// ===========================================================================

template <typename T>
__global__ void softmax_kernel(const T* __restrict__ input,
                                T* __restrict__ output,
                                int batch, int dim) {
    int b = blockIdx.x;
    if (b >= batch) return;

    const T* in_row = input + b * dim;
    T* out_row = output + b * dim;

    // Online softmax: each thread maintains a running (max, sum_exp) pair.
    // When we encounter a new max, we rescale the running sum.
    T thread_max = T(-1e38);
    T thread_sum = T(0);

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        T val = in_row[i];
        if (val > thread_max) {
            thread_sum = thread_sum * fast_exp_dispatch(thread_max - val) + T(1);
            thread_max = val;
        } else {
            thread_sum += fast_exp_dispatch(val - thread_max);
        }
    }

    // Block-wide reduction of (max, sum) pairs via warp shuffles
    __shared__ T s_max[32];
    __shared__ T s_sum[32];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    // Warp-level combine
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        T other_max, other_sum;
        if constexpr (sizeof(T) == 4) {
            other_max = __shfl_down_sync(0xffffffff, thread_max, offset);
            other_sum = __shfl_down_sync(0xffffffff, thread_sum, offset);
        } else {
            other_max = shfl_down_double(thread_max, offset);
            other_sum = shfl_down_double(thread_sum, offset);
        }
        T new_max = max(thread_max, other_max);
        thread_sum = thread_sum * fast_exp_dispatch(thread_max - new_max) +
                     other_sum * fast_exp_dispatch(other_max - new_max);
        thread_max = new_max;
    }

    if (lane == 0) {
        s_max[wid] = thread_max;
        s_sum[wid] = thread_sum;
    }
    __syncthreads();

    // Cross-warp reduction in first warp
    int num_warps = (blockDim.x + 31) / 32;
    if (wid == 0) {
        thread_max = (threadIdx.x < num_warps) ? s_max[threadIdx.x] : T(-1e38);
        thread_sum = (threadIdx.x < num_warps) ? s_sum[threadIdx.x] : T(0);
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            T other_max, other_sum;
            if constexpr (sizeof(T) == 4) {
                other_max = __shfl_down_sync(0xffffffff, thread_max, offset);
                other_sum = __shfl_down_sync(0xffffffff, thread_sum, offset);
            } else {
                other_max = shfl_down_double(thread_max, offset);
                other_sum = shfl_down_double(thread_sum, offset);
            }
            T new_max = max(thread_max, other_max);
            thread_sum = thread_sum * fast_exp_dispatch(thread_max - new_max) +
                         other_sum * fast_exp_dispatch(other_max - new_max);
            thread_max = new_max;
        }
    }

    __shared__ T row_max_s, row_inv_sum_s;
    if (threadIdx.x == 0) {
        row_max_s = thread_max;
        row_inv_sum_s = fast_rcp_dispatch(thread_sum);
    }
    __syncthreads();

    T row_max = row_max_s;
    T inv_sum = row_inv_sum_s;

    constexpr int VW = Vec4<T>::width;
    int vec_dim = dim / VW;
    for (int i = threadIdx.x; i < vec_dim; i += blockDim.x) {
        auto v = Vec4<T>::load(&in_row[i * VW]);
        if constexpr (sizeof(T) == 4) {
            v.x = fast_exp_dispatch(v.x - row_max) * inv_sum;
            v.y = fast_exp_dispatch(v.y - row_max) * inv_sum;
            v.z = fast_exp_dispatch(v.z - row_max) * inv_sum;
            v.w = fast_exp_dispatch(v.w - row_max) * inv_sum;
        } else {
            v.x = fast_exp_dispatch(v.x - row_max) * inv_sum;
            v.y = fast_exp_dispatch(v.y - row_max) * inv_sum;
        }
        Vec4<T>::store(&out_row[i * VW], v);
    }
    for (int i = vec_dim * VW + threadIdx.x; i < dim; i += blockDim.x)
        out_row[i] = fast_exp_dispatch(in_row[i] - row_max) * inv_sum;
}

template <typename T>
void launch_softmax_kernel(const T* input, T* output, int batch, int dim, cudaStream_t stream) {
    int threads = (dim <= 256) ? 128 : DEFAULT_BLOCK_SIZE;
    softmax_kernel<<<batch, threads, 0, stream>>>(input, output, batch, dim);
    CUDEEP_CHECK_LAST_KERNEL();
}

// ===========================================================================
// Fused Warp-Cooperative Cross-Entropy Loss
//
// One warp per sample. Online softmax (single pass over the class dimension
// using warp shuffles to combine running max + sum_exp), then immediately
// compute -log(softmax[target]). No intermediate softmax materialisation,
// no shared memory for the class-dimension reduction.
//
// For batch > warps-per-block, multiple warps share a block and their
// partial losses are block-reduced before atomicAdd.
// ===========================================================================

constexpr int CE_WARPS_PER_BLOCK = 8;
constexpr int CE_THREADS = CE_WARPS_PER_BLOCK * WARP_SIZE;

template <typename T>
__device__ __forceinline__ void warp_online_softmax_reduce(T& my_max, T& my_sum) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        T other_max, other_sum;
        if constexpr (sizeof(T) == 4) {
            other_max = __shfl_down_sync(0xffffffff, my_max, offset);
            other_sum = __shfl_down_sync(0xffffffff, my_sum, offset);
        } else {
            other_max = shfl_down_double(my_max, offset);
            other_sum = shfl_down_double(my_sum, offset);
        }
        T new_max = max(my_max, other_max);
        my_sum = my_sum * fast_exp_dispatch(my_max - new_max) +
                 other_sum * fast_exp_dispatch(other_max - new_max);
        my_max = new_max;
    }
}

template <typename T>
__global__ void cross_entropy_loss_kernel(
    const T* __restrict__ logits, const int* __restrict__ targets, T* loss,
    int batch, int num_classes
) {
    int warp_id_global = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane = threadIdx.x & 31;
    int local_warp = threadIdx.x >> 5;

    __shared__ T s_loss[CE_WARPS_PER_BLOCK];
    s_loss[local_warp] = T(0);
    __syncthreads();

    T warp_loss = T(0);

    for (int sample = warp_id_global; sample < batch;
         sample += (gridDim.x * blockDim.x) / WARP_SIZE) {
        const T* row = logits + sample * num_classes;
        int target = targets[sample];

        T my_max = T(-1e38);
        T my_sum = T(0);

        for (int c = lane; c < num_classes; c += WARP_SIZE) {
            T val = row[c];
            if (val > my_max) {
                my_sum = my_sum * fast_exp_dispatch(my_max - val) + T(1);
                my_max = val;
            } else {
                my_sum += fast_exp_dispatch(val - my_max);
            }
        }

        warp_online_softmax_reduce(my_max, my_sum);

        if (lane == 0) {
            T log_softmax = row[target] - my_max - fast_log_dispatch(my_sum);
            warp_loss -= log_softmax;
        }
    }

    if (lane == 0)
        s_loss[local_warp] = warp_loss;
    __syncthreads();

    if (threadIdx.x < CE_WARPS_PER_BLOCK) {
        T val = s_loss[threadIdx.x];
        val = warp_reduce_sum(static_cast<float>(val));
        if (threadIdx.x == 0)
            atomicAdd(loss, static_cast<T>(val) / static_cast<T>(batch));
    }
}

template <typename T>
void launch_cross_entropy_loss_kernel(
    const T* logits, const int* targets, T* loss,
    int batch, int num_classes, cudaStream_t stream
) {
    CUDEEP_CHECK_CUDA(cudaMemsetAsync(loss, 0, sizeof(T), stream));
    int total_warps = (batch + 0) ;  // one warp per sample ideal
    int blocks = min(ceil_div(total_warps, CE_WARPS_PER_BLOCK), 256);
    blocks = max(blocks, 1);
    cross_entropy_loss_kernel<<<blocks, CE_THREADS, 0, stream>>>(
        logits, targets, loss, batch, num_classes);
    CUDEEP_CHECK_LAST_KERNEL();
}

// ---- Explicit instantiations ----

template void launch_mse_loss_kernel<float>(const float*, const float*, float*, int64_t, cudaStream_t);
template void launch_mse_loss_kernel<double>(const double*, const double*, double*, int64_t, cudaStream_t);
template void launch_cross_entropy_loss_kernel<float>(const float*, const int*, float*, int, int, cudaStream_t);
template void launch_cross_entropy_loss_kernel<double>(const double*, const int*, double*, int, int, cudaStream_t);
template void launch_softmax_kernel<float>(const float*, float*, int, int, cudaStream_t);
template void launch_softmax_kernel<double>(const double*, double*, int, int, cudaStream_t);

}  // namespace kernels
}  // namespace cudeep
