#include "cudeep/kernels/optim.cuh"
#include "cudeep/error.cuh"

#include <cmath>

namespace cudeep {
namespace kernels {

// ===========================================================================
// Vectorized optimizer kernels — float4 loads/stores, grid-stride loop.
//
// Adam/AdamW are especially memory-bound: each element touches param, grad,
// m, v (4 reads + 3 writes). Vectorizing these with float4 roughly doubles
// effective memory bandwidth by issuing 128-bit transactions.
// ===========================================================================

// ---- SGD with momentum + weight decay (float4 vectorized) ----

__global__ void sgd_vec4_kernel(
    float* __restrict__ param, const float* __restrict__ grad, float* __restrict__ velocity,
    int64_t n, float lr, float momentum, float weight_decay
) {
    constexpr int VW = 4;
    int64_t vec_n = n / VW;
    bool has_vel = velocity != nullptr && momentum != 0.0f;
    bool has_wd  = weight_decay != 0.0f;

    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < vec_n;
         i += blockDim.x * gridDim.x) {
        int64_t base = i * VW;
        float4 p = *reinterpret_cast<float4*>(param + base);
        float4 g = *reinterpret_cast<const float4*>(grad + base);

        if (has_wd) {
            g.x += weight_decay * p.x;
            g.y += weight_decay * p.y;
            g.z += weight_decay * p.z;
            g.w += weight_decay * p.w;
        }

        if (has_vel) {
            float4 v = *reinterpret_cast<float4*>(velocity + base);
            v.x = momentum * v.x + g.x;
            v.y = momentum * v.y + g.y;
            v.z = momentum * v.z + g.z;
            v.w = momentum * v.w + g.w;
            *reinterpret_cast<float4*>(velocity + base) = v;
            p.x -= lr * v.x;
            p.y -= lr * v.y;
            p.z -= lr * v.z;
            p.w -= lr * v.w;
        } else {
            p.x -= lr * g.x;
            p.y -= lr * g.y;
            p.z -= lr * g.z;
            p.w -= lr * g.w;
        }
        *reinterpret_cast<float4*>(param + base) = p;
    }

    // Tail
    int64_t tail = vec_n * VW;
    for (int64_t idx = tail + blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += blockDim.x * gridDim.x) {
        float g = grad[idx];
        if (has_wd) g += weight_decay * param[idx];
        if (has_vel) {
            velocity[idx] = momentum * velocity[idx] + g;
            param[idx] -= lr * velocity[idx];
        } else {
            param[idx] -= lr * g;
        }
    }
}

// ---- Adam (float4 vectorized) ----

__global__ void adam_vec4_kernel(
    float* __restrict__ param, const float* __restrict__ grad,
    float* __restrict__ m_buf, float* __restrict__ v_buf,
    int64_t n, float lr, float beta1, float beta2, float eps,
    float weight_decay, float bc1, float bc2
) {
    constexpr int VW = 4;
    int64_t vec_n = n / VW;
    float one_minus_b1 = 1.0f - beta1;
    float one_minus_b2 = 1.0f - beta2;

    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < vec_n;
         i += blockDim.x * gridDim.x) {
        int64_t base = i * VW;
        float4 p = *reinterpret_cast<float4*>(param + base);
        float4 g = *reinterpret_cast<const float4*>(grad + base);
        float4 m = *reinterpret_cast<float4*>(m_buf + base);
        float4 v = *reinterpret_cast<float4*>(v_buf + base);

        #define ADAM_STEP(c) do {                                  \
            float gc = g.c;                                        \
            if (weight_decay != 0.0f) gc += weight_decay * p.c;   \
            m.c = beta1 * m.c + one_minus_b1 * gc;                \
            v.c = beta2 * v.c + one_minus_b2 * gc * gc;           \
            float mh = m.c / bc1;                                  \
            float vh = v.c / bc2;                                  \
            p.c -= lr * mh * rsqrtf(vh + eps);                     \
        } while(0)

        ADAM_STEP(x); ADAM_STEP(y); ADAM_STEP(z); ADAM_STEP(w);
        #undef ADAM_STEP

        *reinterpret_cast<float4*>(param + base) = p;
        *reinterpret_cast<float4*>(m_buf + base) = m;
        *reinterpret_cast<float4*>(v_buf + base) = v;
    }

    int64_t tail = vec_n * VW;
    for (int64_t idx = tail + blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += blockDim.x * gridDim.x) {
        float g = grad[idx];
        if (weight_decay != 0.0f) g += weight_decay * param[idx];
        m_buf[idx] = beta1 * m_buf[idx] + one_minus_b1 * g;
        v_buf[idx] = beta2 * v_buf[idx] + one_minus_b2 * g * g;
        float mh = m_buf[idx] / bc1;
        float vh = v_buf[idx] / bc2;
        param[idx] -= lr * mh * rsqrtf(vh + eps);
    }
}

// ---- AdamW — decoupled weight decay (float4 vectorized) ----

__global__ void adamw_vec4_kernel(
    float* __restrict__ param, const float* __restrict__ grad,
    float* __restrict__ m_buf, float* __restrict__ v_buf,
    int64_t n, float lr, float beta1, float beta2, float eps,
    float weight_decay, float bc1, float bc2
) {
    constexpr int VW = 4;
    int64_t vec_n = n / VW;
    float one_minus_b1 = 1.0f - beta1;
    float one_minus_b2 = 1.0f - beta2;
    float wd_factor = lr * weight_decay;

    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < vec_n;
         i += blockDim.x * gridDim.x) {
        int64_t base = i * VW;
        float4 p = *reinterpret_cast<float4*>(param + base);
        float4 g = *reinterpret_cast<const float4*>(grad + base);
        float4 m = *reinterpret_cast<float4*>(m_buf + base);
        float4 v = *reinterpret_cast<float4*>(v_buf + base);

        // Decoupled weight decay first
        p.x -= wd_factor * p.x;
        p.y -= wd_factor * p.y;
        p.z -= wd_factor * p.z;
        p.w -= wd_factor * p.w;

        #define ADAMW_STEP(c) do {                                 \
            m.c = beta1 * m.c + one_minus_b1 * g.c;               \
            v.c = beta2 * v.c + one_minus_b2 * g.c * g.c;         \
            float mh = m.c / bc1;                                  \
            float vh = v.c / bc2;                                  \
            p.c -= lr * mh * rsqrtf(vh + eps);                     \
        } while(0)

        ADAMW_STEP(x); ADAMW_STEP(y); ADAMW_STEP(z); ADAMW_STEP(w);
        #undef ADAMW_STEP

        *reinterpret_cast<float4*>(param + base) = p;
        *reinterpret_cast<float4*>(m_buf + base) = m;
        *reinterpret_cast<float4*>(v_buf + base) = v;
    }

    int64_t tail = vec_n * VW;
    for (int64_t idx = tail + blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += blockDim.x * gridDim.x) {
        param[idx] -= wd_factor * param[idx];
        float g = grad[idx];
        m_buf[idx] = beta1 * m_buf[idx] + one_minus_b1 * g;
        v_buf[idx] = beta2 * v_buf[idx] + one_minus_b2 * g * g;
        float mh = m_buf[idx] / bc1;
        float vh = v_buf[idx] / bc2;
        param[idx] -= lr * mh * rsqrtf(vh + eps);
    }
}

// ---- Scalar double-precision kernels (fallback) ----

template <typename T>
__global__ void sgd_update_kernel(
    T* param, const T* grad, T* velocity,
    int64_t n, float lr, float momentum, float weight_decay
) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += blockDim.x * gridDim.x) {
        T g = grad[idx];
        if (weight_decay != 0.0f)
            g += static_cast<T>(weight_decay) * param[idx];
        if (momentum != 0.0f && velocity != nullptr) {
            velocity[idx] = static_cast<T>(momentum) * velocity[idx] + g;
            param[idx] -= static_cast<T>(lr) * velocity[idx];
        } else {
            param[idx] -= static_cast<T>(lr) * g;
        }
    }
}

template <typename T>
__global__ void adam_update_kernel(
    T* param, const T* grad, T* m, T* v,
    int64_t n, float lr, float beta1, float beta2, float eps,
    float weight_decay, float bc1, float bc2
) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += blockDim.x * gridDim.x) {
        T g = grad[idx];
        if (weight_decay != 0.0f) g += static_cast<T>(weight_decay) * param[idx];
        m[idx] = static_cast<T>(beta1) * m[idx] + (T(1) - T(beta1)) * g;
        v[idx] = static_cast<T>(beta2) * v[idx] + (T(1) - T(beta2)) * g * g;
        T mh = m[idx] / static_cast<T>(bc1);
        T vh = v[idx] / static_cast<T>(bc2);
        param[idx] -= static_cast<T>(lr) * mh / (sqrt(vh) + static_cast<T>(eps));
    }
}

template <typename T>
__global__ void adamw_update_kernel(
    T* param, const T* grad, T* m, T* v,
    int64_t n, float lr, float beta1, float beta2, float eps,
    float weight_decay, float bc1, float bc2
) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += blockDim.x * gridDim.x) {
        param[idx] -= static_cast<T>(lr * weight_decay) * param[idx];
        T g = grad[idx];
        m[idx] = static_cast<T>(beta1) * m[idx] + (T(1) - T(beta1)) * g;
        v[idx] = static_cast<T>(beta2) * v[idx] + (T(1) - T(beta2)) * g * g;
        T mh = m[idx] / static_cast<T>(bc1);
        T vh = v[idx] / static_cast<T>(bc2);
        param[idx] -= static_cast<T>(lr) * mh / (sqrt(vh) + static_cast<T>(eps));
    }
}

// ---- Launch wrappers ----

static int grid_blocks(int64_t n, int threads) {
    int64_t work = (n + 3) / 4;
    int blocks = static_cast<int>((work + threads - 1) / threads);
    return max(min(blocks, 1024), 1);
}

template <>
void launch_sgd_update_kernel<float>(
    float* param, const float* grad, float* velocity,
    int64_t n, float lr, float momentum, float weight_decay,
    cudaStream_t stream
) {
    sgd_vec4_kernel<<<grid_blocks(n, DEFAULT_BLOCK_SIZE), DEFAULT_BLOCK_SIZE, 0, stream>>>(
        param, grad, velocity, n, lr, momentum, weight_decay);
    CUDEEP_CHECK_LAST_KERNEL();
}

template <>
void launch_sgd_update_kernel<double>(
    double* param, const double* grad, double* velocity,
    int64_t n, float lr, float momentum, float weight_decay,
    cudaStream_t stream
) {
    int blocks = min(ceil_div(static_cast<int>(n), DEFAULT_BLOCK_SIZE), 1024);
    sgd_update_kernel<<<blocks, DEFAULT_BLOCK_SIZE, 0, stream>>>(
        param, grad, velocity, n, lr, momentum, weight_decay);
    CUDEEP_CHECK_LAST_KERNEL();
}

template <>
void launch_adam_update_kernel<float>(
    float* param, const float* grad, float* m, float* v,
    int64_t n, float lr, float beta1, float beta2, float eps,
    float weight_decay, int step,
    cudaStream_t stream
) {
    float bc1 = 1.0f - powf(beta1, static_cast<float>(step));
    float bc2 = 1.0f - powf(beta2, static_cast<float>(step));
    adam_vec4_kernel<<<grid_blocks(n, DEFAULT_BLOCK_SIZE), DEFAULT_BLOCK_SIZE, 0, stream>>>(
        param, grad, m, v, n, lr, beta1, beta2, eps, weight_decay, bc1, bc2);
    CUDEEP_CHECK_LAST_KERNEL();
}

template <>
void launch_adam_update_kernel<double>(
    double* param, const double* grad, double* m, double* v,
    int64_t n, float lr, float beta1, float beta2, float eps,
    float weight_decay, int step,
    cudaStream_t stream
) {
    float bc1 = 1.0f - powf(beta1, static_cast<float>(step));
    float bc2 = 1.0f - powf(beta2, static_cast<float>(step));
    int blocks = min(ceil_div(static_cast<int>(n), DEFAULT_BLOCK_SIZE), 1024);
    adam_update_kernel<<<blocks, DEFAULT_BLOCK_SIZE, 0, stream>>>(
        param, grad, m, v, n, lr, beta1, beta2, eps, weight_decay, bc1, bc2);
    CUDEEP_CHECK_LAST_KERNEL();
}

template <>
void launch_adamw_update_kernel<float>(
    float* param, const float* grad, float* m, float* v,
    int64_t n, float lr, float beta1, float beta2, float eps,
    float weight_decay, int step,
    cudaStream_t stream
) {
    float bc1 = 1.0f - powf(beta1, static_cast<float>(step));
    float bc2 = 1.0f - powf(beta2, static_cast<float>(step));
    adamw_vec4_kernel<<<grid_blocks(n, DEFAULT_BLOCK_SIZE), DEFAULT_BLOCK_SIZE, 0, stream>>>(
        param, grad, m, v, n, lr, beta1, beta2, eps, weight_decay, bc1, bc2);
    CUDEEP_CHECK_LAST_KERNEL();
}

template <>
void launch_adamw_update_kernel<double>(
    double* param, const double* grad, double* m, double* v,
    int64_t n, float lr, float beta1, float beta2, float eps,
    float weight_decay, int step,
    cudaStream_t stream
) {
    float bc1 = 1.0f - powf(beta1, static_cast<float>(step));
    float bc2 = 1.0f - powf(beta2, static_cast<float>(step));
    int blocks = min(ceil_div(static_cast<int>(n), DEFAULT_BLOCK_SIZE), 1024);
    adamw_update_kernel<<<blocks, DEFAULT_BLOCK_SIZE, 0, stream>>>(
        param, grad, m, v, n, lr, beta1, beta2, eps, weight_decay, bc1, bc2);
    CUDEEP_CHECK_LAST_KERNEL();
}

}  // namespace kernels
}  // namespace cudeep
