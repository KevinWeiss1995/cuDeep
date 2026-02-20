#include "cudeep/kernels/reduce.cuh"
#include "cudeep/memory.cuh"
#include "cudeep/error.cuh"

#include <cfloat>

namespace cudeep {
namespace kernels {

// ---- Sum reduction ----

template <typename T>
__global__ void sum_kernel(const T* input, T* output, int64_t n) {
    __shared__ T shared[DEFAULT_BLOCK_SIZE];
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    shared[threadIdx.x] = (idx < n) ? input[idx] : static_cast<T>(0);
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] += shared[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(output, shared[0]);
    }
}

template <typename T>
void launch_sum_kernel(const T* input, T* output, int64_t n, cudaStream_t stream) {
    CUDEEP_CHECK_CUDA(cudaMemsetAsync(output, 0, sizeof(T), stream));
    int threads = DEFAULT_BLOCK_SIZE;
    int blocks = ceil_div(static_cast<int>(n), threads);
    sum_kernel<<<blocks, threads, 0, stream>>>(input, output, n);
    CUDEEP_CHECK_LAST_KERNEL();
}

// ---- Mean reduction ----

template <typename T>
__global__ void divide_scalar_kernel(T* val, T divisor) {
    if (threadIdx.x == 0) {
        *val /= divisor;
    }
}

template <typename T>
void launch_mean_kernel(const T* input, T* output, int64_t n, cudaStream_t stream) {
    if (n == 0) {
        CUDEEP_CHECK_CUDA(cudaMemsetAsync(output, 0, sizeof(T), stream));
        return;
    }
    launch_sum_kernel(input, output, n, stream);
    divide_scalar_kernel<<<1, 1, 0, stream>>>(output, static_cast<T>(n));
    CUDEEP_CHECK_LAST_KERNEL();
}

// ---- Max reduction (two-level) ----

template <typename T>
__global__ void max_reduce_kernel(const T* input, T* block_results, int64_t n) {
    __shared__ T shared[DEFAULT_BLOCK_SIZE];
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    shared[threadIdx.x] = (idx < n) ? input[idx] : static_cast<T>(-1e38);
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] = max(shared[threadIdx.x], shared[threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        block_results[blockIdx.x] = shared[0];
    }
}

template <typename T>
void launch_max_kernel(const T* input, T* output, int64_t n, cudaStream_t stream) {
    int threads = DEFAULT_BLOCK_SIZE;
    int blocks = ceil_div(static_cast<int>(n), threads);

    if (blocks == 1) {
        max_reduce_kernel<<<1, threads, 0, stream>>>(input, output, n);
        CUDEEP_CHECK_LAST_KERNEL();
    } else {
        T* temp = static_cast<T*>(device_malloc(blocks * sizeof(T)));
        max_reduce_kernel<<<blocks, threads, 0, stream>>>(input, temp, n);
        CUDEEP_CHECK_LAST_KERNEL();
        launch_max_kernel(temp, output, static_cast<int64_t>(blocks), stream);
        CUDEEP_CHECK_CUDA(cudaStreamSynchronize(stream));
        device_free(temp);
    }
}

// ---- Min reduction (two-level) ----

template <typename T>
__global__ void min_reduce_kernel(const T* input, T* block_results, int64_t n) {
    __shared__ T shared[DEFAULT_BLOCK_SIZE];
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    shared[threadIdx.x] = (idx < n) ? input[idx] : static_cast<T>(1e38);
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] = min(shared[threadIdx.x], shared[threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        block_results[blockIdx.x] = shared[0];
    }
}

template <typename T>
void launch_min_kernel(const T* input, T* output, int64_t n, cudaStream_t stream) {
    int threads = DEFAULT_BLOCK_SIZE;
    int blocks = ceil_div(static_cast<int>(n), threads);

    if (blocks == 1) {
        min_reduce_kernel<<<1, threads, 0, stream>>>(input, output, n);
        CUDEEP_CHECK_LAST_KERNEL();
    } else {
        T* temp = static_cast<T*>(device_malloc(blocks * sizeof(T)));
        min_reduce_kernel<<<blocks, threads, 0, stream>>>(input, temp, n);
        CUDEEP_CHECK_LAST_KERNEL();
        launch_min_kernel(temp, output, static_cast<int64_t>(blocks), stream);
        CUDEEP_CHECK_CUDA(cudaStreamSynchronize(stream));
        device_free(temp);
    }
}

// ---- Sum along axis ----

template <typename T>
__global__ void sum_along_axis_kernel(
    const T* input, T* output,
    int64_t outer_size, int64_t axis_size, int64_t inner_size
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_out = outer_size * inner_size;
    if (idx >= total_out) return;

    int64_t outer = idx / inner_size;
    int64_t inner = idx % inner_size;

    T acc = static_cast<T>(0);
    for (int64_t a = 0; a < axis_size; ++a) {
        acc += input[(outer * axis_size + a) * inner_size + inner];
    }
    output[idx] = acc;
}

template <typename T>
void launch_sum_along_axis_kernel(
    const T* input, T* output,
    const int64_t* shape, int ndim, int axis,
    cudaStream_t stream
) {
    int64_t outer_size = 1;
    for (int i = 0; i < axis; ++i) outer_size *= shape[i];

    int64_t axis_size = shape[axis];

    int64_t inner_size = 1;
    for (int i = axis + 1; i < ndim; ++i) inner_size *= shape[i];

    int64_t total_out = outer_size * inner_size;
    int threads = DEFAULT_BLOCK_SIZE;
    int blocks = ceil_div(static_cast<int>(total_out), threads);

    sum_along_axis_kernel<<<blocks, threads, 0, stream>>>(
        input, output, outer_size, axis_size, inner_size);
    CUDEEP_CHECK_LAST_KERNEL();
}

// ---- Explicit instantiations ----

template void launch_sum_kernel<float>(const float*, float*, int64_t, cudaStream_t);
template void launch_sum_kernel<double>(const double*, double*, int64_t, cudaStream_t);
template void launch_mean_kernel<float>(const float*, float*, int64_t, cudaStream_t);
template void launch_mean_kernel<double>(const double*, double*, int64_t, cudaStream_t);
template void launch_max_kernel<float>(const float*, float*, int64_t, cudaStream_t);
template void launch_max_kernel<double>(const double*, double*, int64_t, cudaStream_t);
template void launch_min_kernel<float>(const float*, float*, int64_t, cudaStream_t);
template void launch_min_kernel<double>(const double*, double*, int64_t, cudaStream_t);
template void launch_sum_along_axis_kernel<float>(const float*, float*, const int64_t*, int, int, cudaStream_t);
template void launch_sum_along_axis_kernel<double>(const double*, double*, const int64_t*, int, int, cudaStream_t);

}  // namespace kernels
}  // namespace cudeep
