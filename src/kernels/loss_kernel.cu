#include "cudeep/kernels/loss.cuh"
#include "cudeep/error.cuh"

namespace cudeep {
namespace kernels {

// ---- MSE Loss ----

template <typename T>
__global__ void mse_loss_kernel(const T* pred, const T* target, T* loss, int64_t n) {
    __shared__ T shared_sum[DEFAULT_BLOCK_SIZE];
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    T val = static_cast<T>(0);
    if (idx < n) {
        T diff = pred[idx] - target[idx];
        val = diff * diff;
    }

    shared_sum[threadIdx.x] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(loss, shared_sum[0] / static_cast<T>(n));
    }
}

template <typename T>
void launch_mse_loss_kernel(const T* pred, const T* target, T* loss, int64_t n, cudaStream_t stream) {
    CUDEEP_CHECK_CUDA(cudaMemsetAsync(loss, 0, sizeof(T), stream));
    int threads = DEFAULT_BLOCK_SIZE;
    int blocks = ceil_div(static_cast<int>(n), threads);
    mse_loss_kernel<<<blocks, threads, 0, stream>>>(pred, target, loss, n);
    CUDEEP_CHECK_LAST_KERNEL();
}

// ---- Softmax (one block per row, parallel within row) ----

template <typename T>
__global__ void softmax_kernel(const T* input, T* output, int batch, int dim) {
    __shared__ T shared[DEFAULT_BLOCK_SIZE];

    int b = blockIdx.x;
    if (b >= batch) return;

    const T* in_row = input + b * dim;
    T* out_row = output + b * dim;

    T local_max = static_cast<T>(-1e38);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        local_max = max(local_max, in_row[i]);
    }
    shared[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            shared[threadIdx.x] = max(shared[threadIdx.x], shared[threadIdx.x + s]);
        __syncthreads();
    }
    T row_max = shared[0];
    __syncthreads();

    T local_sum = static_cast<T>(0);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        T e = exp(in_row[i] - row_max);
        out_row[i] = e;
        local_sum += e;
    }
    shared[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            shared[threadIdx.x] += shared[threadIdx.x + s];
        __syncthreads();
    }
    T row_sum = shared[0];
    __syncthreads();

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        out_row[i] /= row_sum;
    }
}

template <typename T>
void launch_softmax_kernel(const T* input, T* output, int batch, int dim, cudaStream_t stream) {
    int threads = (dim < DEFAULT_BLOCK_SIZE) ? DEFAULT_BLOCK_SIZE : DEFAULT_BLOCK_SIZE;
    softmax_kernel<<<batch, threads, 0, stream>>>(input, output, batch, dim);
    CUDEEP_CHECK_LAST_KERNEL();
}

// ---- Cross-Entropy Loss ----
// Fused log-softmax + NLL loss, numerically stable

template <typename T>
__global__ void cross_entropy_loss_kernel(
    const T* logits, const int* targets, T* loss,
    int batch, int num_classes
) {
    __shared__ T shared_sum[DEFAULT_BLOCK_SIZE];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    T val = static_cast<T>(0);
    if (idx < batch) {
        const T* row = logits + idx * num_classes;
        int target = targets[idx];

        T max_val = row[0];
        for (int c = 1; c < num_classes; ++c) {
            max_val = max(max_val, row[c]);
        }

        T sum_exp = static_cast<T>(0);
        for (int c = 0; c < num_classes; ++c) {
            sum_exp += exp(row[c] - max_val);
        }

        T log_softmax = row[target] - max_val - log(sum_exp);
        val = -log_softmax;
    }

    shared_sum[threadIdx.x] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(loss, shared_sum[0] / static_cast<T>(batch));
    }
}

template <typename T>
void launch_cross_entropy_loss_kernel(
    const T* logits, const int* targets, T* loss,
    int batch, int num_classes, cudaStream_t stream
) {
    CUDEEP_CHECK_CUDA(cudaMemsetAsync(loss, 0, sizeof(T), stream));
    int threads = DEFAULT_BLOCK_SIZE;
    int blocks = ceil_div(batch, threads);
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
