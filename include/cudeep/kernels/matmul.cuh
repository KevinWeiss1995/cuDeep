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
void launch_matmul_kernel_fp32(
    const T* A, const T* B, T* C,
    int M, int N, int K,
    cudaStream_t stream = nullptr);

template <typename T>
void launch_matmul_tiled_kernel(
    const T* A, const T* B, T* C,
    int M, int N, int K,
    int tile_size,
    cudaStream_t stream = nullptr);

void launch_matmul_kernel_fp16(
    const __half* A, const __half* B, float* C,
    int M, int N, int K,
    cudaStream_t stream = nullptr);

enum class GemmEpilogue {
    None,
    ReLU,
    GELU,
    SiLU,
    Sigmoid
};

void launch_matmul_fused_act(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    GemmEpilogue epilogue,
    cudaStream_t stream = nullptr);

}  // namespace kernels
}  // namespace cudeep
