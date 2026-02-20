#pragma once

#include "../common.cuh"

namespace cudeep {
namespace kernels {

template <typename T>
void launch_neg_kernel(const T* input, T* output, int64_t n, cudaStream_t stream = nullptr);

template <typename T>
void launch_exp_kernel(const T* input, T* output, int64_t n, cudaStream_t stream = nullptr);

template <typename T>
void launch_log_kernel(const T* input, T* output, int64_t n, cudaStream_t stream = nullptr);

template <typename T>
void launch_sqrt_kernel(const T* input, T* output, int64_t n, cudaStream_t stream = nullptr);

template <typename T>
void launch_pow_kernel(const T* input, float exponent, T* output, int64_t n, cudaStream_t stream = nullptr);

template <typename T>
void launch_abs_kernel(const T* input, T* output, int64_t n, cudaStream_t stream = nullptr);

template <typename T>
void launch_clamp_kernel(const T* input, T* output, float lo, float hi, int64_t n, cudaStream_t stream = nullptr);

template <typename T>
void launch_gt_mask_kernel(const T* input, float threshold, T* output, int64_t n, cudaStream_t stream = nullptr);

template <typename T>
void launch_sum_reduce_rows_kernel(
    const T* input, T* output,
    int64_t rows, int64_t cols, cudaStream_t stream = nullptr);

}  // namespace kernels
}  // namespace cudeep
