#pragma once

#include "../common.cuh"

namespace cudeep {
namespace kernels {

template <typename T>
void launch_batchnorm_forward_kernel(
    const T* input, T* output,
    const T* weight, const T* bias,
    T* running_mean, T* running_var,
    int batch, int channels, int spatial,
    float eps, float momentum, bool training,
    cudaStream_t stream = nullptr);

template <typename T>
void launch_layernorm_forward_kernel(
    const T* input, T* output,
    const T* weight, const T* bias,
    int batch_size, int normalized_size, float eps,
    cudaStream_t stream = nullptr);

}  // namespace kernels
}  // namespace cudeep
