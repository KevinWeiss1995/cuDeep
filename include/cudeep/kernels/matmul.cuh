#pragma once

#include "../common.cuh"

namespace cudeep {
namespace kernels {

template <typename T>
void launch_matmul_kernel(
    const T* A, const T* B, T* C,
    int M, int N, int K,
    cudaStream_t stream = nullptr);

template <typename T>
void launch_matmul_tiled_kernel(
    const T* A, const T* B, T* C,
    int M, int N, int K,
    int tile_size,
    cudaStream_t stream = nullptr);

}  // namespace kernels
}  // namespace cudeep
