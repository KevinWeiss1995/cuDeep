#include "cudeep/kernels/norm.cuh"
#include "cudeep/error.cuh"

namespace cudeep {
namespace kernels {

// ---- BatchNorm2d ----
// Phase 1: compute per-channel mean and variance (one block per channel)

template <typename T>
__global__ void batchnorm_stats_kernel(
    const T* input, T* mean, T* var,
    int batch, int channels, int spatial
) {
    int c = blockIdx.x;
    if (c >= channels) return;

    int total = batch * spatial;

    __shared__ T shared[DEFAULT_BLOCK_SIZE];

    T sum = static_cast<T>(0);
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        int b = i / spatial;
        int s = i % spatial;
        sum += input[(b * channels + c) * spatial + s];
    }

    shared[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) shared[threadIdx.x] += shared[threadIdx.x + s];
        __syncthreads();
    }

    T chan_mean = shared[0] / static_cast<T>(total);
    if (threadIdx.x == 0) mean[c] = chan_mean;
    __syncthreads();

    T var_sum = static_cast<T>(0);
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        int b = i / spatial;
        int s = i % spatial;
        T diff = input[(b * channels + c) * spatial + s] - chan_mean;
        var_sum += diff * diff;
    }

    shared[threadIdx.x] = var_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) shared[threadIdx.x] += shared[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0) var[c] = shared[0] / static_cast<T>(total);
}

// Phase 2: normalize + affine

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
    T x_hat = (input[idx] - mean[c]) / sqrt(var[c] + static_cast<T>(eps));
    output[idx] = weight[c] * x_hat + bias[c];
}

// Phase 3 (optional): update running stats

template <typename T>
__global__ void batchnorm_update_running_kernel(
    T* running_mean, T* running_var,
    const T* batch_mean, const T* batch_var,
    int channels, float momentum
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;

    running_mean[c] = static_cast<T>(1.0f - momentum) * running_mean[c] +
                      static_cast<T>(momentum) * batch_mean[c];
    running_var[c]  = static_cast<T>(1.0f - momentum) * running_var[c] +
                      static_cast<T>(momentum) * batch_var[c];
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

// ---- LayerNorm ----
// One block per batch element

template <typename T>
__global__ void layernorm_forward_kernel(
    const T* input, T* output,
    const T* weight, const T* bias,
    int batch_size, int normalized_size, float eps
) {
    int b = blockIdx.x;
    if (b >= batch_size) return;

    const T* in_row = input + b * normalized_size;
    T* out_row = output + b * normalized_size;

    __shared__ T shared[DEFAULT_BLOCK_SIZE];

    T sum = static_cast<T>(0);
    for (int i = threadIdx.x; i < normalized_size; i += blockDim.x) {
        sum += in_row[i];
    }

    shared[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) shared[threadIdx.x] += shared[threadIdx.x + s];
        __syncthreads();
    }
    T mean = shared[0] / static_cast<T>(normalized_size);
    __syncthreads();

    T var_sum = static_cast<T>(0);
    for (int i = threadIdx.x; i < normalized_size; i += blockDim.x) {
        T diff = in_row[i] - mean;
        var_sum += diff * diff;
    }
    shared[threadIdx.x] = var_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) shared[threadIdx.x] += shared[threadIdx.x + s];
        __syncthreads();
    }
    T inv_std = static_cast<T>(1) / sqrt(shared[0] / static_cast<T>(normalized_size) + static_cast<T>(eps));
    __syncthreads();

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
