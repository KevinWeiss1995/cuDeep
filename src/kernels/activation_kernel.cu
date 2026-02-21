#include "cudeep/kernels/activation.cuh"
#include "cudeep/error.cuh"

#include <cmath>

namespace cudeep {
namespace kernels {

// ===========================================================================
// Vectorized activation kernels — float4 loads/stores, grid-stride loop,
// fast math intrinsics (__expf, etc.) for float path.
//
// Previous version: scalar, one element per thread, switch in hot path.
// New version: 4 elements per thread via float4, specialized per activation
//   to eliminate branch in inner loop, fast math where possible.
// ===========================================================================

// ---- Device helpers (using hardware SFU where available) ----

template <typename T>
__device__ __forceinline__ T act_relu(T x) {
    return x > T(0) ? x : T(0);
}

template <typename T>
__device__ __forceinline__ T act_sigmoid(T x) {
    return T(1) / (T(1) + exp(-x));
}
template <>
__device__ __forceinline__ float act_sigmoid<float>(float x) {
    return fast_sigmoid(x);
}

template <typename T>
__device__ __forceinline__ T act_tanh(T x) {
    return tanh(x);
}
template <>
__device__ __forceinline__ float act_tanh<float>(float x) {
    return fast_tanh(x);
}

template <typename T>
__device__ __forceinline__ T act_gelu(T x) {
    constexpr T c = T(0.7978845608);
    constexpr T k = T(0.044715);
    T inner = c * (x + k * x * x * x);
    return T(0.5) * x * (T(1) + tanh(inner));
}
template <>
__device__ __forceinline__ float act_gelu<float>(float x) {
    float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + fast_tanh(inner));
}

template <typename T>
__device__ __forceinline__ T act_silu(T x) {
    return x * act_sigmoid(x);
}

template <typename T>
__device__ __forceinline__ T act_leaky_relu(T x, T alpha) {
    return x > T(0) ? x : alpha * x;
}

// ---- Vectorized forward kernel (float specialization) ----
// Processes 4 elements per thread using float4 loads/stores.

template <typename ActFn>
__global__ void activation_fwd_vec4_kernel(
    const float* __restrict__ input, float* __restrict__ output,
    int64_t n, ActFn fn
) {
    constexpr int VW = 4;
    int64_t vec_n = n / VW;

    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < vec_n;
         i += blockDim.x * gridDim.x) {
        float4 v = *reinterpret_cast<const float4*>(input + i * VW);
        v.x = fn(v.x);
        v.y = fn(v.y);
        v.z = fn(v.z);
        v.w = fn(v.w);
        *reinterpret_cast<float4*>(output + i * VW) = v;
    }

    // Tail elements
    int64_t tail_start = vec_n * VW;
    for (int64_t i = tail_start + threadIdx.x + blockIdx.x * blockDim.x;
         i < n;
         i += blockDim.x * gridDim.x) {
        output[i] = fn(input[i]);
    }
}

// ---- Vectorized backward kernel (float specialization) ----

template <typename BwdFn>
__global__ void activation_bwd_vec4_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    float* __restrict__ grad_input,
    int64_t n, BwdFn fn
) {
    constexpr int VW = 4;
    int64_t vec_n = n / VW;

    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < vec_n;
         i += blockDim.x * gridDim.x) {
        float4 go = *reinterpret_cast<const float4*>(grad_output + i * VW);
        float4 x  = *reinterpret_cast<const float4*>(input + i * VW);
        float4 gi;
        gi.x = fn(go.x, x.x);
        gi.y = fn(go.y, x.y);
        gi.z = fn(go.z, x.z);
        gi.w = fn(go.w, x.w);
        *reinterpret_cast<float4*>(grad_input + i * VW) = gi;
    }

    int64_t tail_start = vec_n * VW;
    for (int64_t i = tail_start + threadIdx.x + blockIdx.x * blockDim.x;
         i < n;
         i += blockDim.x * gridDim.x) {
        grad_input[i] = fn(grad_output[i], input[i]);
    }
}

// ---- Scalar fallback for double ----

template <typename T>
__global__ void activation_forward_scalar_kernel(
    const T* input, T* output, int64_t n,
    ActivationType act_type, float alpha
) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += blockDim.x * gridDim.x) {
        T x = input[idx];
        T result;
        switch (act_type) {
            case ActivationType::ReLU:      result = act_relu(x); break;
            case ActivationType::Sigmoid:   result = act_sigmoid(x); break;
            case ActivationType::Tanh:      result = act_tanh(x); break;
            case ActivationType::GELU:      result = act_gelu(x); break;
            case ActivationType::SiLU:      result = act_silu(x); break;
            case ActivationType::LeakyReLU: result = act_leaky_relu(x, T(alpha)); break;
            default: result = x;
        }
        output[idx] = result;
    }
}

template <typename T>
__global__ void activation_backward_scalar_kernel(
    const T* grad_output, const T* input, T* grad_input,
    int64_t n, ActivationType act_type, float alpha
) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += blockDim.x * gridDim.x) {
        T x = input[idx];
        T go = grad_output[idx];
        T grad;
        switch (act_type) {
            case ActivationType::ReLU:
                grad = x > T(0) ? go : T(0); break;
            case ActivationType::Sigmoid: {
                T s = act_sigmoid(x);
                grad = go * s * (T(1) - s); break;
            }
            case ActivationType::Tanh: {
                T t = act_tanh(x);
                grad = go * (T(1) - t * t); break;
            }
            case ActivationType::GELU: {
                constexpr T c = T(0.7978845608);
                constexpr T k = T(0.044715);
                T x3 = x * x * x;
                T inner = c * (x + k * x3);
                T th = tanh(inner);
                T sech2 = T(1) - th * th;
                T d_inner = c * (T(1) + T(3) * k * x * x);
                grad = go * (T(0.5) * (T(1) + th) + T(0.5) * x * sech2 * d_inner);
                break;
            }
            case ActivationType::SiLU: {
                T s = act_sigmoid(x);
                grad = go * (s * (T(1) + x * (T(1) - s)));
                break;
            }
            case ActivationType::LeakyReLU:
                grad = x > T(0) ? go : T(alpha) * go; break;
            default:
                grad = go;
        }
        grad_input[idx] = grad;
    }
}

// ---- Launch wrappers ----

static int grid_blocks(int64_t n, int threads) {
    int64_t work = (n + 3) / 4;  // at least 1 if n > 0
    int blocks = static_cast<int>((work + threads - 1) / threads);
    return max(min(blocks, 1024), 1);
}

template <>
void launch_activation_forward_kernel<float>(
    const float* input, float* output, int64_t n,
    ActivationType act_type, float alpha, cudaStream_t stream
) {
    int threads = DEFAULT_BLOCK_SIZE;
    int blocks = grid_blocks(n, threads);

    switch (act_type) {
        case ActivationType::ReLU:
            activation_fwd_vec4_kernel<<<blocks, threads, 0, stream>>>(
                input, output, n, [] __device__ (float x) { return act_relu(x); });
            break;
        case ActivationType::Sigmoid:
            activation_fwd_vec4_kernel<<<blocks, threads, 0, stream>>>(
                input, output, n, [] __device__ (float x) { return act_sigmoid(x); });
            break;
        case ActivationType::Tanh:
            activation_fwd_vec4_kernel<<<blocks, threads, 0, stream>>>(
                input, output, n, [] __device__ (float x) { return act_tanh(x); });
            break;
        case ActivationType::GELU:
            activation_fwd_vec4_kernel<<<blocks, threads, 0, stream>>>(
                input, output, n, [] __device__ (float x) { return act_gelu(x); });
            break;
        case ActivationType::SiLU:
            activation_fwd_vec4_kernel<<<blocks, threads, 0, stream>>>(
                input, output, n, [] __device__ (float x) { return act_silu(x); });
            break;
        case ActivationType::LeakyReLU: {
            float a = alpha;
            activation_fwd_vec4_kernel<<<blocks, threads, 0, stream>>>(
                input, output, n, [a] __device__ (float x) { return act_leaky_relu(x, a); });
            break;
        }
    }
    CUDEEP_CHECK_LAST_KERNEL();
}

template <>
void launch_activation_forward_kernel<double>(
    const double* input, double* output, int64_t n,
    ActivationType act_type, float alpha, cudaStream_t stream
) {
    int threads = DEFAULT_BLOCK_SIZE;
    int blocks = min(ceil_div(static_cast<int>(n), threads), 1024);
    activation_forward_scalar_kernel<<<blocks, threads, 0, stream>>>(input, output, n, act_type, alpha);
    CUDEEP_CHECK_LAST_KERNEL();
}

template <>
void launch_activation_backward_kernel<float>(
    const float* grad_output, const float* input, float* grad_input,
    int64_t n, ActivationType act_type, float alpha, cudaStream_t stream
) {
    int threads = DEFAULT_BLOCK_SIZE;
    int blocks = grid_blocks(n, threads);

    switch (act_type) {
        case ActivationType::ReLU:
            activation_bwd_vec4_kernel<<<blocks, threads, 0, stream>>>(
                grad_output, input, grad_input, n,
                [] __device__ (float go, float x) -> float {
                    return x > 0.0f ? go : 0.0f;
                });
            break;
        case ActivationType::Sigmoid:
            activation_bwd_vec4_kernel<<<blocks, threads, 0, stream>>>(
                grad_output, input, grad_input, n,
                [] __device__ (float go, float x) -> float {
                    float s = fast_sigmoid(x);
                    return go * s * (1.0f - s);
                });
            break;
        case ActivationType::Tanh:
            activation_bwd_vec4_kernel<<<blocks, threads, 0, stream>>>(
                grad_output, input, grad_input, n,
                [] __device__ (float go, float x) -> float {
                    float t = fast_tanh(x);
                    return go * (1.0f - t * t);
                });
            break;
        case ActivationType::GELU:
            activation_bwd_vec4_kernel<<<blocks, threads, 0, stream>>>(
                grad_output, input, grad_input, n,
                [] __device__ (float go, float x) -> float {
                    float x3 = x * x * x;
                    float inner = 0.7978845608f * (x + 0.044715f * x3);
                    float th = fast_tanh(inner);
                    float sech2 = 1.0f - th * th;
                    float d_inner = 0.7978845608f * (1.0f + 3.0f * 0.044715f * x * x);
                    return go * (0.5f * (1.0f + th) + 0.5f * x * sech2 * d_inner);
                });
            break;
        case ActivationType::SiLU:
            activation_bwd_vec4_kernel<<<blocks, threads, 0, stream>>>(
                grad_output, input, grad_input, n,
                [] __device__ (float go, float x) -> float {
                    float s = fast_sigmoid(x);
                    return go * (s * (1.0f + x * (1.0f - s)));
                });
            break;
        case ActivationType::LeakyReLU: {
            float a = alpha;
            activation_bwd_vec4_kernel<<<blocks, threads, 0, stream>>>(
                grad_output, input, grad_input, n,
                [a] __device__ (float go, float x) -> float {
                    return x > 0.0f ? go : a * go;
                });
            break;
        }
    }
    CUDEEP_CHECK_LAST_KERNEL();
}

template <>
void launch_activation_backward_kernel<double>(
    const double* grad_output, const double* input, double* grad_input,
    int64_t n, ActivationType act_type, float alpha, cudaStream_t stream
) {
    int threads = DEFAULT_BLOCK_SIZE;
    int blocks = min(ceil_div(static_cast<int>(n), threads), 1024);
    activation_backward_scalar_kernel<<<blocks, threads, 0, stream>>>(grad_output, input, grad_input, n, act_type, alpha);
    CUDEEP_CHECK_LAST_KERNEL();
}

}  // namespace kernels
}  // namespace cudeep
