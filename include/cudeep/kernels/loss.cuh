#pragma once

#include "../common.cuh"

namespace cudeep {
namespace kernels {

template <typename T>
void launch_mse_loss_kernel(
    const T* pred, const T* target, T* loss,
    int64_t n,
    cudaStream_t stream = nullptr);

template <typename T>
void launch_cross_entropy_loss_kernel(
    const T* logits, const int* targets, T* loss,
    int batch, int num_classes,
    cudaStream_t stream = nullptr);

template <typename T>
void launch_softmax_kernel(
    const T* input, T* output,
    int batch, int dim,
    cudaStream_t stream = nullptr);

}  // namespace kernels
}  // namespace cudeep
