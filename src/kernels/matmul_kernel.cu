#include "cudeep/kernels/matmul.cuh"
#include "cudeep/error.cuh"

namespace cudeep {
namespace kernels {

// ---------------------------------------------------------------------------
// Register-blocked tiled SGEMM
//
// Each thread block computes a BM×BN tile of C.
// Each thread computes a TM×TN sub-tile using register accumulators.
// Shared memory tiles are loaded collaboratively and reused BK times.
//
// Config: BM=64, BN=64, BK=16, TM=4, TN=4
//   → 256 threads/block, 16 register accumulators per thread
//   → 8 KB shared memory per K-tile step
// ---------------------------------------------------------------------------

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;
constexpr int TM = 4;
constexpr int TN = 4;
constexpr int THREADS_M = BM / TM;  // 16
constexpr int THREADS_N = BN / TN;  // 16
constexpr int NUM_THREADS = THREADS_M * THREADS_N;  // 256

template <typename T>
__global__ void matmul_kernel(const T* __restrict__ A,
                              const T* __restrict__ B,
                              T* __restrict__ C,
                              int M, int N, int K) {
    __shared__ T As[BM][BK + 1];
    __shared__ T Bs[BK][BN + 1];

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x % THREADS_N;
    const int ty = threadIdx.x / THREADS_N;

    const int aRowBase = by * BM;
    const int bColBase = bx * BN;

    T acc[TM][TN] = {};

    constexpr int LOAD_A_PER_THREAD = (BM * BK) / NUM_THREADS;  // 4
    constexpr int LOAD_B_PER_THREAD = (BK * BN) / NUM_THREADS;  // 4

    for (int bk = 0; bk < K; bk += BK) {

        #pragma unroll
        for (int i = 0; i < LOAD_A_PER_THREAD; ++i) {
            int linear = threadIdx.x + i * NUM_THREADS;
            int r = linear / BK;
            int c = linear % BK;
            int gRow = aRowBase + r;
            int gCol = bk + c;
            As[r][c] = (gRow < M && gCol < K) ? A[gRow * K + gCol] : T(0);
        }

        #pragma unroll
        for (int i = 0; i < LOAD_B_PER_THREAD; ++i) {
            int linear = threadIdx.x + i * NUM_THREADS;
            int r = linear / BN;
            int c = linear % BN;
            int gRow = bk + r;
            int gCol = bColBase + c;
            Bs[r][c] = (gRow < K && gCol < N) ? B[gRow * N + gCol] : T(0);
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            T a_reg[TM];
            T b_reg[TN];

            #pragma unroll
            for (int m = 0; m < TM; ++m)
                a_reg[m] = As[ty * TM + m][k];

            #pragma unroll
            for (int n = 0; n < TN; ++n)
                b_reg[n] = Bs[k][tx * TN + n];

            #pragma unroll
            for (int m = 0; m < TM; ++m)
                #pragma unroll
                for (int n = 0; n < TN; ++n)
                    acc[m][n] += a_reg[m] * b_reg[n];
        }

        __syncthreads();
    }

    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        int gRow = aRowBase + ty * TM + m;
        if (gRow >= M) continue;
        #pragma unroll
        for (int n = 0; n < TN; ++n) {
            int gCol = bColBase + tx * TN + n;
            if (gCol < N)
                C[gRow * N + gCol] = acc[m][n];
        }
    }
}

// Keep old tiled dim for API compat
constexpr int TILE_DIM = 32;

template <typename T>
void launch_matmul_kernel(const T* A, const T* B, T* C, int M, int N, int K, cudaStream_t stream) {
    dim3 threads(NUM_THREADS);
    dim3 blocks(ceil_div(N, BN), ceil_div(M, BM));
    matmul_kernel<<<blocks, threads, 0, stream>>>(A, B, C, M, N, K);
    CUDEEP_CHECK_LAST_KERNEL();
}

template <typename T>
void launch_matmul_tiled_kernel(const T* A, const T* B, T* C, int M, int N, int K, int /*tile_size*/, cudaStream_t stream) {
    launch_matmul_kernel(A, B, C, M, N, K, stream);
}

template void launch_matmul_kernel<float>(const float*, const float*, float*, int, int, int, cudaStream_t);
template void launch_matmul_kernel<double>(const double*, const double*, double*, int, int, int, cudaStream_t);
template void launch_matmul_tiled_kernel<float>(const float*, const float*, float*, int, int, int, int, cudaStream_t);
template void launch_matmul_tiled_kernel<double>(const double*, const double*, double*, int, int, int, int, cudaStream_t);

}  // namespace kernels
}  // namespace cudeep
