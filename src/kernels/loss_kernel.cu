#include "cudeep/kernels/loss.cuh"
#include "cudeep/error.cuh"

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
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x) {
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
            thread_sum = thread_sum * exp(thread_max - val) + T(1);
            thread_max = val;
        } else {
            thread_sum += exp(val - thread_max);
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
        thread_sum = thread_sum * exp(thread_max - new_max) +
                     other_sum * exp(other_max - new_max);
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
            thread_sum = thread_sum * exp(thread_max - new_max) +
                         other_sum * exp(other_max - new_max);
            thread_max = new_max;
        }
    }

    // Broadcast final max and inv_sum to all threads
    __shared__ T row_max_s, row_inv_sum_s;
    if (threadIdx.x == 0) {
        row_max_s = thread_max;
        row_inv_sum_s = T(1) / thread_sum;
    }
    __syncthreads();

    T row_max = row_max_s;
    T inv_sum = row_inv_sum_s;

    // Pass 2: write normalized values
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        out_row[i] = exp(in_row[i] - row_max) * inv_sum;
}

template <typename T>
void launch_softmax_kernel(const T* input, T* output, int batch, int dim, cudaStream_t stream) {
    int threads = (dim <= 256) ? 128 : DEFAULT_BLOCK_SIZE;
    softmax_kernel<<<batch, threads, 0, stream>>>(input, output, batch, dim);
    CUDEEP_CHECK_LAST_KERNEL();
}

// ===========================================================================
// Cross-Entropy Loss — parallelise over batch with block-level reduction
// Use online softmax trick per row for numerical stability.
// ===========================================================================

template <typename T>
__global__ void cross_entropy_loss_kernel(
    const T* __restrict__ logits, const int* __restrict__ targets, T* loss,
    int batch, int num_classes
) {
    T acc = T(0);
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < batch;
         idx += blockDim.x * gridDim.x) {
        const T* row = logits + idx * num_classes;
        int target = targets[idx];

        T row_max = row[0];
        for (int c = 1; c < num_classes; ++c)
            row_max = max(row_max, row[c]);

        T sum_exp = T(0);
        for (int c = 0; c < num_classes; ++c)
            sum_exp += exp(row[c] - row_max);

        T log_softmax = row[target] - row_max - log(sum_exp);
        acc -= log_softmax;
    }

    acc = block_reduce_sum(acc);

    if (threadIdx.x == 0)
        atomicAdd(loss, acc / static_cast<T>(batch));
}

template <typename T>
void launch_cross_entropy_loss_kernel(
    const T* logits, const int* targets, T* loss,
    int batch, int num_classes, cudaStream_t stream
) {
    CUDEEP_CHECK_CUDA(cudaMemsetAsync(loss, 0, sizeof(T), stream));
    int threads = DEFAULT_BLOCK_SIZE;
    int blocks = min(ceil_div(batch, threads), 256);
    cross_entropy_loss_kernel<<<blocks, threads, 0, stream>>>(
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
