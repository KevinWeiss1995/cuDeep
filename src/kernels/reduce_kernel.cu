#include "cudeep/kernels/reduce.cuh"
#include "cudeep/error.cuh"

namespace cudeep {
namespace kernels {

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

template <typename T>
void launch_mean_kernel(const T* input, T* output, int64_t n, cudaStream_t stream) {
    launch_sum_kernel(input, output, n, stream);
    // TODO: divide by n on device
}

template <typename T>
void launch_max_kernel(const T* input, T* output, int64_t n, cudaStream_t stream) {
    // TODO: implement parallel max reduction
}

template <typename T>
void launch_min_kernel(const T* input, T* output, int64_t n, cudaStream_t stream) {
    // TODO: implement parallel min reduction
}

template <typename T>
void launch_sum_along_axis_kernel(
    const T* input, T* output,
    const int64_t* shape, int ndim, int axis,
    cudaStream_t stream
) {
    // TODO: implement axis-wise reduction
}

template void launch_sum_kernel<float>(const float*, float*, int64_t, cudaStream_t);
template void launch_sum_kernel<double>(const double*, double*, int64_t, cudaStream_t);
template void launch_mean_kernel<float>(const float*, float*, int64_t, cudaStream_t);
template void launch_mean_kernel<double>(const double*, double*, int64_t, cudaStream_t);
template void launch_max_kernel<float>(const float*, float*, int64_t, cudaStream_t);
template void launch_max_kernel<double>(const double*, double*, int64_t, cudaStream_t);
template void launch_min_kernel<float>(const float*, float*, int64_t, cudaStream_t);
template void launch_min_kernel<double>(const double*, double*, int64_t, cudaStream_t);

}  // namespace kernels
}  // namespace cudeep
