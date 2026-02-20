#include "cudeep/kernels/matmul.cuh"
#include "cudeep/error.cuh"

namespace cudeep {
namespace kernels {

constexpr int TILE_DIM = 32;

template <typename T>
__global__ void matmul_naive_kernel(const T* A, const T* B, T* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        T acc = static_cast<T>(0);
        for (int k = 0; k < K; ++k) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}

template <typename T>
__global__ void matmul_tiled_kernel(const T* A, const T* B, T* C, int M, int N, int K) {
    __shared__ T tileA[TILE_DIM][TILE_DIM];
    __shared__ T tileB[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    T acc = static_cast<T>(0);

    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; ++t) {
        int tileCol = t * TILE_DIM + threadIdx.x;
        int tileRow = t * TILE_DIM + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] = (row < M && tileCol < K) ? A[row * K + tileCol] : static_cast<T>(0);
        tileB[threadIdx.y][threadIdx.x] = (tileRow < K && col < N) ? B[tileRow * N + col] : static_cast<T>(0);

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            acc += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

template <typename T>
void launch_matmul_kernel(const T* A, const T* B, T* C, int M, int N, int K, cudaStream_t stream) {
    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 blocks(ceil_div(N, TILE_DIM), ceil_div(M, TILE_DIM));
    matmul_tiled_kernel<<<blocks, threads, 0, stream>>>(A, B, C, M, N, K);
    CUDEEP_CHECK_LAST_KERNEL();
}

template <typename T>
void launch_matmul_tiled_kernel(const T* A, const T* B, T* C, int M, int N, int K, int tile_size, cudaStream_t stream) {
    (void)tile_size;
    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 blocks(ceil_div(N, TILE_DIM), ceil_div(M, TILE_DIM));
    matmul_tiled_kernel<<<blocks, threads, 0, stream>>>(A, B, C, M, N, K);
    CUDEEP_CHECK_LAST_KERNEL();
}

template void launch_matmul_kernel<float>(const float*, const float*, float*, int, int, int, cudaStream_t);
template void launch_matmul_kernel<double>(const double*, const double*, double*, int, int, int, cudaStream_t);
template void launch_matmul_tiled_kernel<float>(const float*, const float*, float*, int, int, int, int, cudaStream_t);
template void launch_matmul_tiled_kernel<double>(const double*, const double*, double*, int, int, int, int, cudaStream_t);

}  // namespace kernels
}  // namespace cudeep
