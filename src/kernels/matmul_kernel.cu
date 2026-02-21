#include "cudeep/kernels/matmul.cuh"
#include "cudeep/error.cuh"

namespace cudeep {
namespace kernels {

// ===========================================================================
// SGEMM v4 — instruction-issue-rate optimised
//
// The v3 kernel's bottleneck was NOT occupancy (2 blocks/SM with 124 regs,
// 17 KB smem out of 164 KB available). It was **FMA dilution**: 16 scalar
// shared-memory loads per k-step drowned the 64 FMAs (80% FMA ratio). With
// warp scheduler overheads the effective FMA utilisation dropped to ~30%.
//
// Fixes applied:
//   1. float4 smem reads — 2 loads per fragment instead of 8 → 94% FMA ratio
//   2. BK=16 — halves tile count, halves __syncthreads / global-load phases
//   3. cp.async for B — global→shared without register detour (SM 8.0+)
//   4. Fully unrolled k-loop — zero loop overhead
//   5. Fast path skips all boundary checks for aligned matrices
// ===========================================================================

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 16;
constexpr int TM = 8;
constexpr int TN = 8;
constexpr int NUM_THREADS = 256;
constexpr int SMEM_A_STRIDE = BM + 4;   // 132 — avoids bank conflicts on A reads
constexpr int SMEM_B_STRIDE = BN + 4;   // 132

// A: 128×16 = 2048 elems, 256 threads → 8 scalar loads/thread
// B: 16×128 = 2048 elems = 512 float4s, 256 threads → 2 float4 loads/thread

// Shared memory per buffer:
//   As: 16 × 132 × 4 = 8448  bytes
//   Bs: 16 × 132 × 4 = 8448  bytes
//   Total per buffer: 16896 bytes
//   Double-buffered: 33792 bytes ≈ 33 KB — fits 2 blocks in 164 KB/SM

// ---- Inline helpers for cp.async (B loads) ----

__device__ __forceinline__ void cp_async_f4_or_direct(
    float* smem_dst, const float* global_src) {
#if __CUDA_ARCH__ >= 800
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(global_src) : "memory");
#else
    *reinterpret_cast<float4*>(smem_dst) =
        *reinterpret_cast<const float4*>(global_src);
#endif
}

__device__ __forceinline__ void cp_async_fence() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
}

__device__ __forceinline__ void cp_async_wait() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_all;\n" ::: "memory");
#endif
}

// ---- Fast-path: M%BM==0, N%BN==0, K%BK==0, N%4==0 ----

__global__ __launch_bounds__(NUM_THREADS, 2)
void sgemm_fast_kernel(const float* __restrict__ A,
                       const float* __restrict__ B,
                       float* __restrict__ C,
                       int M, int N, int K) {

    __shared__ float As[2][BK][SMEM_A_STRIDE];
    __shared__ float Bs[2][BK][SMEM_B_STRIDE];

    const int bm = blockIdx.y * BM;
    const int bn = blockIdx.x * BN;
    const int tx = threadIdx.x % (BN / TN);     // 0..15
    const int ty = threadIdx.x / (BN / TN);     // 0..15
    const int tid = threadIdx.x;
    const int a_base = ty * TM;                  // always multiple of 8
    const int b_base = tx * TN;                  // always multiple of 8

    // A load mapping — scalar, column-major for perfect coalescing
    const int a_col = tid % BK;                  // 0..15
    const int a_row0 = tid / BK;                 // 0..15
    // 8 loads per thread at a_row0 + i*16 for i=0..7

    // B load mapping — 2 float4s per thread
    const int b_f4_id0 = tid;                    // 0..255
    const int b_row0 = b_f4_id0 / (BN / 4);     // tid / 32
    const int b_col4_0 = b_f4_id0 % (BN / 4);   // tid % 32
    const int b_f4_id1 = tid + NUM_THREADS;      // 256..511
    const int b_row1 = b_f4_id1 / (BN / 4);     // 8 + tid/32
    const int b_col4_1 = b_f4_id1 % (BN / 4);   // tid % 32

    float acc[TM][TN] = {};

    // ---- Tile loading macros ----
    #define LOAD_A_TILE(buf, bk_off) do {                                   \
        const float* _Ab = A + bm * K + (bk_off) + a_col;                  \
        _Pragma("unroll")                                                    \
        for (int _s = 0; _s < 8; ++_s)                                     \
            As[buf][a_col][a_row0 + _s * 16] = _Ab[(a_row0 + _s * 16) * K]; \
    } while(0)

    #define LOAD_B_TILE(buf, bk_off) do {                                   \
        cp_async_f4_or_direct(                                               \
            &Bs[buf][b_row0][b_col4_0 * 4],                                 \
            &B[((bk_off) + b_row0) * N + bn + b_col4_0 * 4]);              \
        cp_async_f4_or_direct(                                               \
            &Bs[buf][b_row1][b_col4_1 * 4],                                 \
            &B[((bk_off) + b_row1) * N + bn + b_col4_1 * 4]);              \
    } while(0)

    // Compute one tile with float4 smem reads — 4 loads + 64 FMAs per k-step
    #define COMPUTE_TILE(buf) do {                                          \
        float4 _a4_lo, _a4_hi, _b4_lo, _b4_hi;                            \
        float _af[TM], _bf[TN];                                            \
        _Pragma("unroll")                                                    \
        for (int _k = 0; _k < BK; ++_k) {                                 \
            _a4_lo = *reinterpret_cast<const float4*>(                      \
                         &As[buf][_k][a_base]);                             \
            _a4_hi = *reinterpret_cast<const float4*>(                      \
                         &As[buf][_k][a_base + 4]);                         \
            _af[0] = _a4_lo.x; _af[1] = _a4_lo.y;                         \
            _af[2] = _a4_lo.z; _af[3] = _a4_lo.w;                         \
            _af[4] = _a4_hi.x; _af[5] = _a4_hi.y;                         \
            _af[6] = _a4_hi.z; _af[7] = _a4_hi.w;                         \
            _b4_lo = *reinterpret_cast<const float4*>(                      \
                         &Bs[buf][_k][b_base]);                             \
            _b4_hi = *reinterpret_cast<const float4*>(                      \
                         &Bs[buf][_k][b_base + 4]);                         \
            _bf[0] = _b4_lo.x; _bf[1] = _b4_lo.y;                         \
            _bf[2] = _b4_lo.z; _bf[3] = _b4_lo.w;                         \
            _bf[4] = _b4_hi.x; _bf[5] = _b4_hi.y;                         \
            _bf[6] = _b4_hi.z; _bf[7] = _b4_hi.w;                         \
            _Pragma("unroll")                                                \
            for (int _m = 0; _m < TM; ++_m)                                \
                _Pragma("unroll")                                            \
                for (int _n = 0; _n < TN; ++_n)                            \
                    acc[_m][_n] += _af[_m] * _bf[_n];                       \
        }                                                                    \
    } while(0)

    // ---- Main loop with double buffering ----
    LOAD_A_TILE(0, 0);
    LOAD_B_TILE(0, 0);
    cp_async_fence();
    cp_async_wait();
    __syncthreads();

    const int num_tiles = K / BK;
    int buf = 0;

    for (int t = 0; t < num_tiles - 1; ++t) {
        LOAD_A_TILE(1 - buf, (t + 1) * BK);
        LOAD_B_TILE(1 - buf, (t + 1) * BK);
        cp_async_fence();

        COMPUTE_TILE(buf);

        cp_async_wait();
        buf = 1 - buf;
        __syncthreads();
    }
    COMPUTE_TILE(buf);

    #undef LOAD_A_TILE
    #undef LOAD_B_TILE
    #undef COMPUTE_TILE

    // Vectorised store (guaranteed aligned in fast path)
    float* Crow = C + (bm + a_base) * N + bn + b_base;
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        *reinterpret_cast<float4*>(Crow)     = make_float4(acc[m][0], acc[m][1], acc[m][2], acc[m][3]);
        *reinterpret_cast<float4*>(Crow + 4) = make_float4(acc[m][4], acc[m][5], acc[m][6], acc[m][7]);
        Crow += N;
    }
}

// ---- General kernel: handles arbitrary M, N, K ----

__global__ __launch_bounds__(NUM_THREADS, 2)
void sgemm_general_kernel(const float* __restrict__ A,
                          const float* __restrict__ B,
                          float* __restrict__ C,
                          int M, int N, int K) {

    __shared__ float As[2][BK][SMEM_A_STRIDE];
    __shared__ float Bs[2][BK][SMEM_B_STRIDE];

    const int bm = blockIdx.y * BM;
    const int bn = blockIdx.x * BN;
    const int tx = threadIdx.x % (BN / TN);
    const int ty = threadIdx.x / (BN / TN);
    const int tid = threadIdx.x;
    const int a_base = ty * TM;
    const int b_base = tx * TN;

    const int a_col = tid % BK;
    const int a_row0 = tid / BK;
    const int b_f4_id0 = tid;
    const int b_row0 = b_f4_id0 / (BN / 4);
    const int b_col4_0 = b_f4_id0 % (BN / 4);
    const int b_f4_id1 = tid + NUM_THREADS;
    const int b_row1 = b_f4_id1 / (BN / 4);
    const int b_col4_1 = b_f4_id1 % (BN / 4);

    float acc[TM][TN] = {};

    auto load_A_safe = [&](int buf_idx, int bk) {
        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            int row = a_row0 + s * 16;
            int gr = bm + row;
            int gc = bk + a_col;
            As[buf_idx][a_col][row] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
    };

    auto load_B_safe = [&](int buf_idx, int bk) {
        // Load 0
        {
            int gr = bk + b_row0;
            int gc = bn + b_col4_0 * 4;
            if (gr < K && gc + 3 < N) {
                int addr = gr * N + gc;
                if ((addr & 3) == 0) {
                    *reinterpret_cast<float4*>(&Bs[buf_idx][b_row0][b_col4_0 * 4]) =
                        *reinterpret_cast<const float4*>(&B[addr]);
                } else {
                    Bs[buf_idx][b_row0][b_col4_0*4+0] = B[addr+0];
                    Bs[buf_idx][b_row0][b_col4_0*4+1] = B[addr+1];
                    Bs[buf_idx][b_row0][b_col4_0*4+2] = B[addr+2];
                    Bs[buf_idx][b_row0][b_col4_0*4+3] = B[addr+3];
                }
            } else {
                for (int j = 0; j < 4; ++j) {
                    int c = gc + j;
                    Bs[buf_idx][b_row0][b_col4_0*4+j] =
                        (gr < K && c < N) ? B[gr * N + c] : 0.0f;
                }
            }
        }
        // Load 1
        {
            int gr = bk + b_row1;
            int gc = bn + b_col4_1 * 4;
            if (gr < K && gc + 3 < N) {
                int addr = gr * N + gc;
                if ((addr & 3) == 0) {
                    *reinterpret_cast<float4*>(&Bs[buf_idx][b_row1][b_col4_1 * 4]) =
                        *reinterpret_cast<const float4*>(&B[addr]);
                } else {
                    Bs[buf_idx][b_row1][b_col4_1*4+0] = B[addr+0];
                    Bs[buf_idx][b_row1][b_col4_1*4+1] = B[addr+1];
                    Bs[buf_idx][b_row1][b_col4_1*4+2] = B[addr+2];
                    Bs[buf_idx][b_row1][b_col4_1*4+3] = B[addr+3];
                }
            } else {
                for (int j = 0; j < 4; ++j) {
                    int c = gc + j;
                    Bs[buf_idx][b_row1][b_col4_1*4+j] =
                        (gr < K && c < N) ? B[gr * N + c] : 0.0f;
                }
            }
        }
    };

    load_A_safe(0, 0);
    load_B_safe(0, 0);
    __syncthreads();

    int num_tiles = (K + BK - 1) / BK;
    int buf = 0;

    for (int t = 0; t < num_tiles - 1; ++t) {
        load_A_safe(1 - buf, (t + 1) * BK);
        load_B_safe(1 - buf, (t + 1) * BK);

        // Compute with float4 smem reads
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float4 a4lo = *reinterpret_cast<const float4*>(&As[buf][k][a_base]);
            float4 a4hi = *reinterpret_cast<const float4*>(&As[buf][k][a_base+4]);
            float4 b4lo = *reinterpret_cast<const float4*>(&Bs[buf][k][b_base]);
            float4 b4hi = *reinterpret_cast<const float4*>(&Bs[buf][k][b_base+4]);
            float af[TM] = {a4lo.x, a4lo.y, a4lo.z, a4lo.w,
                            a4hi.x, a4hi.y, a4hi.z, a4hi.w};
            float bf[TN] = {b4lo.x, b4lo.y, b4lo.z, b4lo.w,
                            b4hi.x, b4hi.y, b4hi.z, b4hi.w};
            #pragma unroll
            for (int m = 0; m < TM; ++m)
                #pragma unroll
                for (int n = 0; n < TN; ++n)
                    acc[m][n] += af[m] * bf[n];
        }

        buf = 1 - buf;
        __syncthreads();
    }

    // Last tile
    #pragma unroll
    for (int k = 0; k < BK; ++k) {
        float4 a4lo = *reinterpret_cast<const float4*>(&As[buf][k][a_base]);
        float4 a4hi = *reinterpret_cast<const float4*>(&As[buf][k][a_base+4]);
        float4 b4lo = *reinterpret_cast<const float4*>(&Bs[buf][k][b_base]);
        float4 b4hi = *reinterpret_cast<const float4*>(&Bs[buf][k][b_base+4]);
        float af[TM] = {a4lo.x, a4lo.y, a4lo.z, a4lo.w,
                        a4hi.x, a4hi.y, a4hi.z, a4hi.w};
        float bf[TN] = {b4lo.x, b4lo.y, b4lo.z, b4lo.w,
                        b4hi.x, b4hi.y, b4hi.z, b4hi.w};
        #pragma unroll
        for (int m = 0; m < TM; ++m)
            #pragma unroll
            for (int n = 0; n < TN; ++n)
                acc[m][n] += af[m] * bf[n];
    }

    // Store
    const int c_row = bm + a_base;
    const int c_col = bn + b_base;
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        int grow = c_row + m;
        if (grow >= M) continue;
        if (c_col + TN <= N && ((grow * N + c_col) & 3) == 0) {
            *reinterpret_cast<float4*>(&C[grow * N + c_col]) =
                make_float4(acc[m][0], acc[m][1], acc[m][2], acc[m][3]);
            *reinterpret_cast<float4*>(&C[grow * N + c_col + 4]) =
                make_float4(acc[m][4], acc[m][5], acc[m][6], acc[m][7]);
        } else {
            #pragma unroll
            for (int n = 0; n < TN; ++n) {
                int gcol = c_col + n;
                if (gcol < N) C[grow * N + gcol] = acc[m][n];
            }
        }
    }
}

// ===========================================================================
// Double-precision fallback (single-buffered, smaller tiles)
// ===========================================================================

constexpr int BM_D = 64, BN_D = 64, BK_D = 16, TM_D = 4, TN_D = 4;
constexpr int NUM_THREADS_D = (BM_D / TM_D) * (BN_D / TN_D);

template <typename T>
__global__ __launch_bounds__(NUM_THREADS_D)
void sgemm_double_kernel(const T* __restrict__ A, const T* __restrict__ B,
                         T* __restrict__ C, int M, int N, int K) {
    __shared__ T As_d[BK_D][BM_D + 4];
    __shared__ T Bs_d[BK_D][BN_D + 4];
    const int bm = blockIdx.y * BM_D, bn = blockIdx.x * BN_D;
    const int tx = threadIdx.x % (BN_D / TN_D), ty = threadIdx.x / (BN_D / TN_D);
    const int tid = threadIdx.x;
    T acc[TM_D][TN_D] = {};
    constexpr int LA = (BM_D * BK_D) / NUM_THREADS_D;
    constexpr int LB = (BK_D * BN_D) / NUM_THREADS_D;
    for (int bk = 0; bk < K; bk += BK_D) {
        for (int i = 0; i < LA; ++i) {
            int lin = tid + i * NUM_THREADS_D;
            int r = lin / BK_D, c = lin % BK_D;
            int gr = bm + r, gc = bk + c;
            As_d[c][r] = (gr < M && gc < K) ? A[gr * K + gc] : T(0);
        }
        for (int i = 0; i < LB; ++i) {
            int lin = tid + i * NUM_THREADS_D;
            int r = lin / BN_D, c = lin % BN_D;
            int gr = bk + r, gc = bn + c;
            Bs_d[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : T(0);
        }
        __syncthreads();
        T af[TM_D], bf[TN_D];
        #pragma unroll
        for (int k = 0; k < BK_D; ++k) {
            for (int m = 0; m < TM_D; ++m) af[m] = As_d[k][ty * TM_D + m];
            for (int n = 0; n < TN_D; ++n) bf[n] = Bs_d[k][tx * TN_D + n];
            for (int m = 0; m < TM_D; ++m)
                for (int n = 0; n < TN_D; ++n)
                    acc[m][n] += af[m] * bf[n];
        }
        __syncthreads();
    }
    for (int m = 0; m < TM_D; ++m) {
        int grow = bm + ty * TM_D + m;
        if (grow >= M) continue;
        for (int n = 0; n < TN_D; ++n) {
            int gcol = bn + tx * TN_D + n;
            if (gcol < N) C[grow * N + gcol] = acc[m][n];
        }
    }
}

// ===========================================================================
// Launch wrappers
// ===========================================================================

template <>
void launch_matmul_kernel<float>(const float* A, const float* B, float* C,
                                  int M, int N, int K, cudaStream_t stream) {
    dim3 blocks(ceil_div(N, BN), ceil_div(M, BM));
    bool fast = (M % BM == 0) && (N % BN == 0) && (K % BK == 0) && (N % 4 == 0);
    if (fast) {
        sgemm_fast_kernel<<<blocks, NUM_THREADS, 0, stream>>>(A, B, C, M, N, K);
    } else {
        sgemm_general_kernel<<<blocks, NUM_THREADS, 0, stream>>>(A, B, C, M, N, K);
    }
    CUDEEP_CHECK_LAST_KERNEL();
}

template <>
void launch_matmul_kernel<double>(const double* A, const double* B, double* C,
                                   int M, int N, int K, cudaStream_t stream) {
    dim3 blocks(ceil_div(N, BN_D), ceil_div(M, BM_D));
    sgemm_double_kernel<<<blocks, NUM_THREADS_D, 0, stream>>>(A, B, C, M, N, K);
    CUDEEP_CHECK_LAST_KERNEL();
}

template <typename T>
void launch_matmul_tiled_kernel(const T* A, const T* B, T* C,
                                 int M, int N, int K, int, cudaStream_t stream) {
    launch_matmul_kernel(A, B, C, M, N, K, stream);
}

template void launch_matmul_kernel<float>(const float*, const float*, float*, int, int, int, cudaStream_t);
template void launch_matmul_kernel<double>(const double*, const double*, double*, int, int, int, cudaStream_t);
template void launch_matmul_tiled_kernel<float>(const float*, const float*, float*, int, int, int, int, cudaStream_t);
template void launch_matmul_tiled_kernel<double>(const double*, const double*, double*, int, int, int, int, cudaStream_t);

}  // namespace kernels
}  // namespace cudeep
