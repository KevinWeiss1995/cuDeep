#pragma once

#include "../common.cuh"

namespace cudeep {
namespace kernels {

template <typename T>
void launch_sum_kernel(const T* input, T* output, int64_t n, cudaStream_t stream = nullptr);

template <typename T>
void launch_mean_kernel(const T* input, T* output, int64_t n, cudaStream_t stream = nullptr);

template <typename T>
void launch_max_kernel(const T* input, T* output, int64_t n, cudaStream_t stream = nullptr);

template <typename T>
void launch_min_kernel(const T* input, T* output, int64_t n, cudaStream_t stream = nullptr);

template <typename T>
void launch_sum_along_axis_kernel(
    const T* input, T* output,
    const int64_t* shape, int ndim, int axis,
    cudaStream_t stream = nullptr);

}  // namespace kernels
}  // namespace cudeep
