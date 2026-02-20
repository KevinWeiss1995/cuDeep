#pragma once

#include "../common.cuh"

namespace cudeep {
namespace kernels {

template <typename T>
void launch_sgd_update_kernel(
    T* param, const T* grad, T* velocity,
    int64_t n, float lr, float momentum, float weight_decay,
    cudaStream_t stream = nullptr);

template <typename T>
void launch_adam_update_kernel(
    T* param, const T* grad, T* m, T* v,
    int64_t n, float lr, float beta1, float beta2, float eps,
    float weight_decay, int step,
    cudaStream_t stream = nullptr);

template <typename T>
void launch_adamw_update_kernel(
    T* param, const T* grad, T* m, T* v,
    int64_t n, float lr, float beta1, float beta2, float eps,
    float weight_decay, int step,
    cudaStream_t stream = nullptr);

}  // namespace kernels
}  // namespace cudeep
