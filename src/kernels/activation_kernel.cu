#include "cudeep/kernels/activation.cuh"
#include "cudeep/error.cuh"

#include <cmath>

namespace cudeep {
namespace kernels {

template <typename T>
__device__ T relu_device(T x) {
    return x > static_cast<T>(0) ? x : static_cast<T>(0);
}

template <typename T>
__device__ T sigmoid_device(T x) {
    return static_cast<T>(1) / (static_cast<T>(1) + exp(-x));
}

template <typename T>
__device__ T gelu_device(T x) {
    constexpr T c = static_cast<T>(0.7978845608);  // sqrt(2/pi)
    constexpr T k = static_cast<T>(0.044715);
    T inner = c * (x + k * x * x * x);
    return static_cast<T>(0.5) * x * (static_cast<T>(1) + tanh(inner));
}

template <typename T>
__device__ T silu_device(T x) {
    return x * sigmoid_device(x);
}

template <typename T>
__global__ void activation_forward_kernel(
    const T* input, T* output, int64_t n,
    ActivationType act_type, float alpha
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    T x = input[idx];
    T result;

    switch (act_type) {
        case ActivationType::ReLU:
            result = relu_device(x);
            break;
        case ActivationType::Sigmoid:
            result = sigmoid_device(x);
            break;
        case ActivationType::Tanh:
            result = tanh(x);
            break;
        case ActivationType::GELU:
            result = gelu_device(x);
            break;
        case ActivationType::SiLU:
            result = silu_device(x);
            break;
        case ActivationType::LeakyReLU:
            result = x > static_cast<T>(0) ? x : static_cast<T>(alpha) * x;
            break;
        default:
            result = x;
    }

    output[idx] = result;
}

template <typename T>
__global__ void activation_backward_kernel(
    const T* grad_output, const T* input, T* grad_input,
    int64_t n, ActivationType act_type, float alpha
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    T x = input[idx];
    T go = grad_output[idx];
    T grad;

    switch (act_type) {
        case ActivationType::ReLU:
            grad = x > static_cast<T>(0) ? go : static_cast<T>(0);
            break;
        case ActivationType::Sigmoid: {
            T s = sigmoid_device(x);
            grad = go * s * (static_cast<T>(1) - s);
            break;
        }
        case ActivationType::Tanh: {
            T t = tanh(x);
            grad = go * (static_cast<T>(1) - t * t);
            break;
        }
        case ActivationType::LeakyReLU:
            grad = x > static_cast<T>(0) ? go : static_cast<T>(alpha) * go;
            break;
        default:
            grad = go;
    }

    grad_input[idx] = grad;
}

template <typename T>
void launch_activation_forward_kernel(
    const T* input, T* output, int64_t n,
    ActivationType act_type, float alpha, cudaStream_t stream
) {
    int threads = DEFAULT_BLOCK_SIZE;
    int blocks = ceil_div(static_cast<int>(n), threads);
    activation_forward_kernel<<<blocks, threads, 0, stream>>>(input, output, n, act_type, alpha);
    CUDEEP_CHECK_LAST_KERNEL();
}

template <typename T>
void launch_activation_backward_kernel(
    const T* grad_output, const T* input, T* grad_input,
    int64_t n, ActivationType act_type, float alpha, cudaStream_t stream
) {
    int threads = DEFAULT_BLOCK_SIZE;
    int blocks = ceil_div(static_cast<int>(n), threads);
    activation_backward_kernel<<<blocks, threads, 0, stream>>>(grad_output, input, grad_input, n, act_type, alpha);
    CUDEEP_CHECK_LAST_KERNEL();
}

template void launch_activation_forward_kernel<float>(const float*, float*, int64_t, ActivationType, float, cudaStream_t);
template void launch_activation_forward_kernel<double>(const double*, double*, int64_t, ActivationType, float, cudaStream_t);
template void launch_activation_backward_kernel<float>(const float*, const float*, float*, int64_t, ActivationType, float, cudaStream_t);
template void launch_activation_backward_kernel<double>(const double*, const double*, double*, int64_t, ActivationType, float, cudaStream_t);

}  // namespace kernels
}  // namespace cudeep
