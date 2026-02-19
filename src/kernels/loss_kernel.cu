#include "cudeep/kernels/loss.cuh"
#include "cudeep/error.cuh"

namespace cudeep {
namespace kernels {

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
__global__ void softmax_kernel(const T* input, T* output, int batch, int dim) {
    int b = blockIdx.x;
    if (b >= batch) return;

    const T* in_row = input + b * dim;
    T* out_row = output + b * dim;

    T max_val = in_row[0];
    for (int i = 1; i < dim; ++i) {
        max_val = max(max_val, in_row[i]);
    }

    T sum_exp = static_cast<T>(0);
    for (int i = 0; i < dim; ++i) {
        out_row[i] = exp(in_row[i] - max_val);
        sum_exp += out_row[i];
    }

    for (int i = 0; i < dim; ++i) {
        out_row[i] /= sum_exp;
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

template <typename T>
void launch_cross_entropy_loss_kernel(
    const T* logits, const int* targets, T* loss,
    int batch, int num_classes, cudaStream_t stream
) {
    // TODO: implement cross-entropy loss kernel
}

template <typename T>
void launch_softmax_kernel(const T* input, T* output, int batch, int dim, cudaStream_t stream) {
    softmax_kernel<<<batch, 1, 0, stream>>>(input, output, batch, dim);
    CUDEEP_CHECK_LAST_KERNEL();
}

template void launch_mse_loss_kernel<float>(const float*, const float*, float*, int64_t, cudaStream_t);
template void launch_mse_loss_kernel<double>(const double*, const double*, double*, int64_t, cudaStream_t);
template void launch_cross_entropy_loss_kernel<float>(const float*, const int*, float*, int, int, cudaStream_t);
template void launch_cross_entropy_loss_kernel<double>(const double*, const int*, double*, int, int, cudaStream_t);
template void launch_softmax_kernel<float>(const float*, float*, int, int, cudaStream_t);
template void launch_softmax_kernel<double>(const double*, double*, int, int, cudaStream_t);

}  // namespace kernels
}  // namespace cudeep
