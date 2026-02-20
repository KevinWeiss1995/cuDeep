#include "cudeep/kernels/loss.cuh"
#include "cudeep/error.cuh"

namespace cudeep {
namespace kernels {

// ---------------------------------------------------------------------------
// MSE Loss — grid-stride + warp-shuffle reduction
// ---------------------------------------------------------------------------

template <typename T>
__global__ void mse_loss_kernel(const T* pred, const T* target, T* loss, int64_t n) {
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

// ---------------------------------------------------------------------------
// Softmax — one block per row, warp-shuffle reductions
// Two passes: (1) find max + accumulate exp, (2) normalize.
// ---------------------------------------------------------------------------

template <typename T>
__global__ void softmax_kernel(const T* input, T* output, int batch, int dim) {
    int b = blockIdx.x;
    if (b >= batch) return;

    const T* in_row = input + b * dim;
    T* out_row = output + b * dim;

    // Pass 1: find row max
    T local_max = T(-1e38);
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        local_max = max(local_max, in_row[i]);
    T row_max = block_reduce_max(local_max);
    row_max = __shfl_sync(0xffffffff, row_max, 0);

    // Pass 2: compute exp(x - max) and accumulate sum
    T local_sum = T(0);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        T e = exp(in_row[i] - row_max);
        out_row[i] = e;
        local_sum += e;
    }
    T row_sum = block_reduce_sum(local_sum);
    row_sum = __shfl_sync(0xffffffff, row_sum, 0);

    // Pass 3: normalize
    T inv_sum = T(1) / row_sum;
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        out_row[i] *= inv_sum;
}

template <typename T>
void launch_softmax_kernel(const T* input, T* output, int batch, int dim, cudaStream_t stream) {
    int threads = DEFAULT_BLOCK_SIZE;
    softmax_kernel<<<batch, threads, 0, stream>>>(input, output, batch, dim);
    CUDEEP_CHECK_LAST_KERNEL();
}

// ---------------------------------------------------------------------------
// Cross-Entropy Loss — fused log-softmax + NLL, grid-stride reduction
// ---------------------------------------------------------------------------

template <typename T>
__global__ void cross_entropy_loss_kernel(
    const T* logits, const int* targets, T* loss,
    int batch, int num_classes
) {
    T acc = T(0);
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < batch;
         idx += blockDim.x * gridDim.x) {
        const T* row = logits + idx * num_classes;
        int target = targets[idx];

        T max_val = row[0];
        for (int c = 1; c < num_classes; ++c)
            max_val = max(max_val, row[c]);

        T sum_exp = T(0);
        for (int c = 0; c < num_classes; ++c)
            sum_exp += exp(row[c] - max_val);

        T log_softmax = row[target] - max_val - log(sum_exp);
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
