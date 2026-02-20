#include "cudeep/kernels/optim.cuh"
#include "cudeep/error.cuh"

#include <cmath>

namespace cudeep {
namespace kernels {

// ---- SGD with optional momentum and weight decay ----

template <typename T>
__global__ void sgd_update_kernel(
    T* param, const T* grad, T* velocity,
    int64_t n, float lr, float momentum, float weight_decay
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    T g = grad[idx];
    if (weight_decay != 0.0f) {
        g += static_cast<T>(weight_decay) * param[idx];
    }

    if (momentum != 0.0f && velocity != nullptr) {
        velocity[idx] = static_cast<T>(momentum) * velocity[idx] + g;
        param[idx] -= static_cast<T>(lr) * velocity[idx];
    } else {
        param[idx] -= static_cast<T>(lr) * g;
    }
}

template <typename T>
void launch_sgd_update_kernel(
    T* param, const T* grad, T* velocity,
    int64_t n, float lr, float momentum, float weight_decay,
    cudaStream_t stream
) {
    int threads = DEFAULT_BLOCK_SIZE;
    int blocks = ceil_div(static_cast<int>(n), threads);
    sgd_update_kernel<<<blocks, threads, 0, stream>>>(
        param, grad, velocity, n, lr, momentum, weight_decay);
    CUDEEP_CHECK_LAST_KERNEL();
}

// ---- Adam ----

template <typename T>
__global__ void adam_update_kernel(
    T* param, const T* grad, T* m, T* v,
    int64_t n, float lr, float beta1, float beta2, float eps,
    float weight_decay, float bias_correction1, float bias_correction2
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    T g = grad[idx];
    if (weight_decay != 0.0f) {
        g += static_cast<T>(weight_decay) * param[idx];
    }

    m[idx] = static_cast<T>(beta1) * m[idx] + (static_cast<T>(1) - static_cast<T>(beta1)) * g;
    v[idx] = static_cast<T>(beta2) * v[idx] + (static_cast<T>(1) - static_cast<T>(beta2)) * g * g;

    T m_hat = m[idx] / static_cast<T>(bias_correction1);
    T v_hat = v[idx] / static_cast<T>(bias_correction2);

    param[idx] -= static_cast<T>(lr) * m_hat / (sqrt(v_hat) + static_cast<T>(eps));
}

template <typename T>
void launch_adam_update_kernel(
    T* param, const T* grad, T* m, T* v,
    int64_t n, float lr, float beta1, float beta2, float eps,
    float weight_decay, int step,
    cudaStream_t stream
) {
    float bias_correction1 = 1.0f - powf(beta1, static_cast<float>(step));
    float bias_correction2 = 1.0f - powf(beta2, static_cast<float>(step));

    int threads = DEFAULT_BLOCK_SIZE;
    int blocks = ceil_div(static_cast<int>(n), threads);
    adam_update_kernel<<<blocks, threads, 0, stream>>>(
        param, grad, m, v, n, lr, beta1, beta2, eps,
        weight_decay, bias_correction1, bias_correction2);
    CUDEEP_CHECK_LAST_KERNEL();
}

// ---- AdamW (decoupled weight decay) ----

template <typename T>
__global__ void adamw_update_kernel(
    T* param, const T* grad, T* m, T* v,
    int64_t n, float lr, float beta1, float beta2, float eps,
    float weight_decay, float bias_correction1, float bias_correction2
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    param[idx] -= static_cast<T>(lr * weight_decay) * param[idx];

    T g = grad[idx];
    m[idx] = static_cast<T>(beta1) * m[idx] + (static_cast<T>(1) - static_cast<T>(beta1)) * g;
    v[idx] = static_cast<T>(beta2) * v[idx] + (static_cast<T>(1) - static_cast<T>(beta2)) * g * g;

    T m_hat = m[idx] / static_cast<T>(bias_correction1);
    T v_hat = v[idx] / static_cast<T>(bias_correction2);

    param[idx] -= static_cast<T>(lr) * m_hat / (sqrt(v_hat) + static_cast<T>(eps));
}

template <typename T>
void launch_adamw_update_kernel(
    T* param, const T* grad, T* m, T* v,
    int64_t n, float lr, float beta1, float beta2, float eps,
    float weight_decay, int step,
    cudaStream_t stream
) {
    float bias_correction1 = 1.0f - powf(beta1, static_cast<float>(step));
    float bias_correction2 = 1.0f - powf(beta2, static_cast<float>(step));

    int threads = DEFAULT_BLOCK_SIZE;
    int blocks = ceil_div(static_cast<int>(n), threads);
    adamw_update_kernel<<<blocks, threads, 0, stream>>>(
        param, grad, m, v, n, lr, beta1, beta2, eps,
        weight_decay, bias_correction1, bias_correction2);
    CUDEEP_CHECK_LAST_KERNEL();
}

// ---- Explicit instantiations ----

template void launch_sgd_update_kernel<float>(float*, const float*, float*, int64_t, float, float, float, cudaStream_t);
template void launch_sgd_update_kernel<double>(double*, const double*, double*, int64_t, float, float, float, cudaStream_t);
template void launch_adam_update_kernel<float>(float*, const float*, float*, float*, int64_t, float, float, float, float, float, int, cudaStream_t);
template void launch_adam_update_kernel<double>(double*, const double*, double*, double*, int64_t, float, float, float, float, float, int, cudaStream_t);
template void launch_adamw_update_kernel<float>(float*, const float*, float*, float*, int64_t, float, float, float, float, float, int, cudaStream_t);
template void launch_adamw_update_kernel<double>(double*, const double*, double*, double*, int64_t, float, float, float, float, float, int, cudaStream_t);

}  // namespace kernels
}  // namespace cudeep
