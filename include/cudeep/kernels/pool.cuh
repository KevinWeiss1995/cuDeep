#pragma once

#include "../common.cuh"

namespace cudeep {
namespace kernels {

template <typename T>
void launch_maxpool2d_forward_kernel(
    const T* input, T* output,
    int batch, int channels,
    int in_h, int in_w,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    cudaStream_t stream = nullptr);

template <typename T>
void launch_avgpool2d_forward_kernel(
    const T* input, T* output,
    int batch, int channels,
    int in_h, int in_w,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    cudaStream_t stream = nullptr);

}  // namespace kernels
}  // namespace cudeep
