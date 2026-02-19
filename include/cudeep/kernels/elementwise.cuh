#pragma once

#include "../common.cuh"

namespace cudeep {
namespace kernels {

template <typename T>
void launch_add_kernel(const T* a, const T* b, T* out, int64_t n, cudaStream_t stream = nullptr);

template <typename T>
void launch_sub_kernel(const T* a, const T* b, T* out, int64_t n, cudaStream_t stream = nullptr);

template <typename T>
void launch_mul_kernel(const T* a, const T* b, T* out, int64_t n, cudaStream_t stream = nullptr);

template <typename T>
void launch_div_kernel(const T* a, const T* b, T* out, int64_t n, cudaStream_t stream = nullptr);

template <typename T>
void launch_scalar_mul_kernel(const T* a, float scalar, T* out, int64_t n, cudaStream_t stream = nullptr);

template <typename T>
void launch_fill_kernel(T* data, T value, int64_t n, cudaStream_t stream = nullptr);

}  // namespace kernels
}  // namespace cudeep
