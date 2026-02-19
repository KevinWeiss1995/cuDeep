#pragma once

#include "../common.cuh"

namespace cudeep {
namespace kernels {

enum class ActivationType {
    ReLU,
    Sigmoid,
    Tanh,
    GELU,
    SiLU,
    LeakyReLU
};

template <typename T>
void launch_activation_forward_kernel(
    const T* input, T* output,
    int64_t n,
    ActivationType act_type,
    float alpha = 0.01f,
    cudaStream_t stream = nullptr);

template <typename T>
void launch_activation_backward_kernel(
    const T* grad_output, const T* input, T* grad_input,
    int64_t n,
    ActivationType act_type,
    float alpha = 0.01f,
    cudaStream_t stream = nullptr);

}  // namespace kernels
}  // namespace cudeep
