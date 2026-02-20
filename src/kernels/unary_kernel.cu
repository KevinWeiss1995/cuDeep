#include "cudeep/kernels/unary.cuh"
#include "cudeep/error.cuh"

#include <cmath>

namespace cudeep {
namespace kernels {

template <typename T>
__global__ void neg_kernel(const T* input, T* output, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = -input[idx];
}

template <typename T>
__global__ void exp_kernel(const T* input, T* output, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = exp(input[idx]);
}

template <typename T>
__global__ void log_kernel(const T* input, T* output, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = log(input[idx]);
}

template <typename T>
__global__ void sqrt_kernel(const T* input, T* output, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = sqrt(input[idx]);
}

template <typename T>
__global__ void pow_kernel(const T* input, float exponent, T* output, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = pow(input[idx], static_cast<T>(exponent));
}

template <typename T>
__global__ void abs_kernel(const T* input, T* output, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = (input[idx] >= static_cast<T>(0)) ? input[idx] : -input[idx];
}

template <typename T>
__global__ void clamp_kernel(const T* input, T* output, float lo, float hi, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        T val = input[idx];
        T lo_t = static_cast<T>(lo);
        T hi_t = static_cast<T>(hi);
        output[idx] = (val < lo_t) ? lo_t : ((val > hi_t) ? hi_t : val);
    }
}

template <typename T>
__global__ void gt_mask_kernel(const T* input, float threshold, T* output, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = (input[idx] > static_cast<T>(threshold))
                      ? static_cast<T>(1) : static_cast<T>(0);
    }
}

template <typename T>
__global__ void sum_reduce_rows_kernel(const T* input, T* output, int64_t rows, int64_t cols) {
    int64_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) return;
    T acc = static_cast<T>(0);
    for (int64_t r = 0; r < rows; ++r) {
        acc += input[r * cols + col];
    }
    output[col] = acc;
}

// ---- Launch wrappers ----

template <typename T>
void launch_neg_kernel(const T* input, T* output, int64_t n, cudaStream_t stream) {
    int threads = DEFAULT_BLOCK_SIZE;
    int blocks = ceil_div(static_cast<int>(n), threads);
    neg_kernel<<<blocks, threads, 0, stream>>>(input, output, n);
    CUDEEP_CHECK_LAST_KERNEL();
}

template <typename T>
void launch_exp_kernel(const T* input, T* output, int64_t n, cudaStream_t stream) {
    int threads = DEFAULT_BLOCK_SIZE;
    int blocks = ceil_div(static_cast<int>(n), threads);
    exp_kernel<<<blocks, threads, 0, stream>>>(input, output, n);
    CUDEEP_CHECK_LAST_KERNEL();
}

template <typename T>
void launch_log_kernel(const T* input, T* output, int64_t n, cudaStream_t stream) {
    int threads = DEFAULT_BLOCK_SIZE;
    int blocks = ceil_div(static_cast<int>(n), threads);
    log_kernel<<<blocks, threads, 0, stream>>>(input, output, n);
    CUDEEP_CHECK_LAST_KERNEL();
}

template <typename T>
void launch_sqrt_kernel(const T* input, T* output, int64_t n, cudaStream_t stream) {
    int threads = DEFAULT_BLOCK_SIZE;
    int blocks = ceil_div(static_cast<int>(n), threads);
    sqrt_kernel<<<blocks, threads, 0, stream>>>(input, output, n);
    CUDEEP_CHECK_LAST_KERNEL();
}

template <typename T>
void launch_abs_kernel(const T* input, T* output, int64_t n, cudaStream_t stream) {
    int threads = DEFAULT_BLOCK_SIZE;
    int blocks = ceil_div(static_cast<int>(n), threads);
    abs_kernel<<<blocks, threads, 0, stream>>>(input, output, n);
    CUDEEP_CHECK_LAST_KERNEL();
}

template <typename T>
void launch_pow_kernel(const T* input, float exponent, T* output, int64_t n, cudaStream_t stream) {
    int threads = DEFAULT_BLOCK_SIZE;
    int blocks = ceil_div(static_cast<int>(n), threads);
    pow_kernel<<<blocks, threads, 0, stream>>>(input, exponent, output, n);
    CUDEEP_CHECK_LAST_KERNEL();
}

template <typename T>
void launch_clamp_kernel(const T* input, T* output, float lo, float hi, int64_t n, cudaStream_t stream) {
    int threads = DEFAULT_BLOCK_SIZE;
    int blocks = ceil_div(static_cast<int>(n), threads);
    clamp_kernel<<<blocks, threads, 0, stream>>>(input, output, lo, hi, n);
    CUDEEP_CHECK_LAST_KERNEL();
}

template <typename T>
void launch_gt_mask_kernel(const T* input, float threshold, T* output, int64_t n, cudaStream_t stream) {
    int threads = DEFAULT_BLOCK_SIZE;
    int blocks = ceil_div(static_cast<int>(n), threads);
    gt_mask_kernel<<<blocks, threads, 0, stream>>>(input, threshold, output, n);
    CUDEEP_CHECK_LAST_KERNEL();
}

template <typename T>
void launch_sum_reduce_rows_kernel(const T* input, T* output,
                                    int64_t rows, int64_t cols, cudaStream_t stream) {
    int threads = DEFAULT_BLOCK_SIZE;
    int blocks = ceil_div(static_cast<int>(cols), threads);
    sum_reduce_rows_kernel<<<blocks, threads, 0, stream>>>(input, output, rows, cols);
    CUDEEP_CHECK_LAST_KERNEL();
}

// ---- Explicit instantiations ----

#define INST_UNARY(name) \
template void launch_##name##_kernel<float>(const float*, float*, int64_t, cudaStream_t); \
template void launch_##name##_kernel<double>(const double*, double*, int64_t, cudaStream_t);

INST_UNARY(neg)
INST_UNARY(exp)
INST_UNARY(log)
INST_UNARY(sqrt)
INST_UNARY(abs)

#undef INST_UNARY

template void launch_pow_kernel<float>(const float*, float, float*, int64_t, cudaStream_t);
template void launch_pow_kernel<double>(const double*, float, double*, int64_t, cudaStream_t);
template void launch_clamp_kernel<float>(const float*, float*, float, float, int64_t, cudaStream_t);
template void launch_clamp_kernel<double>(const double*, double*, float, float, int64_t, cudaStream_t);
template void launch_gt_mask_kernel<float>(const float*, float, float*, int64_t, cudaStream_t);
template void launch_gt_mask_kernel<double>(const double*, float, double*, int64_t, cudaStream_t);
template void launch_sum_reduce_rows_kernel<float>(const float*, float*, int64_t, int64_t, cudaStream_t);
template void launch_sum_reduce_rows_kernel<double>(const double*, double*, int64_t, int64_t, cudaStream_t);

}  // namespace kernels
}  // namespace cudeep
