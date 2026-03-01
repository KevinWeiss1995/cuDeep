#include "cudeep/kernels/matmul.cuh"
#include "cudeep/ptx_intrinsics.cuh"
#include "cudeep/error.cuh"

namespace cudeep {
namespace kernels {

// ===========================================================================
//  SGEMM kernel suite — three paths in descending priority:
//
//    1. TF32 Tensor Core  (SM 8.0+, aligned dims, ~2x throughput of FP32)
//    2. FP32 CUDA Core    (any SM, fast path for aligned dims)
//    3. FP32 CUDA Core    (any SM, general path with boundary checks)
//
//  All share the same tile/smem layout so the loading code is identical.
// ===========================================================================

// ---- Shared tile parameters ----
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 16;
constexpr int NUM_THREADS = 256;
constexpr int SMEM_A_STRIDE = BM + 4;   // 132
constexpr int SMEM_B_STRIDE = BN + 4;   // 132

// ---- FP32 CUDA-core parameters ----
constexpr int TM = 8;
constexpr int TN = 8;

// ---- TC (Tensor Core) parameters ----
constexpr int TC_WM = 32;                       // warp tile M
constexpr int TC_WN = 64;                       // warp tile N
constexpr int TC_WARPS_M = BM / TC_WM;          // 4
constexpr int TC_WARPS_N = BN / TC_WN;          // 2
constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 8;
constexpr int MMA_TILES_M = TC_WM / MMA_M;      // 2
constexpr int MMA_TILES_N = TC_WN / MMA_N;      // 8

// ---- Tile loading params (common to all float kernels) ----
// A: 128×16 = 2048 elems, 256 threads, 8 scalar loads/thread
// B: 16×128 = 2048 elems = 512 float4s, 256 threads, 2 float4/thread

// ===========================================================================
//  Path 1: TF32 Tensor Core SGEMM  (SM 8.0+, aligned dims)
//
//  Each warp computes a 32×64 output tile using mma.sync.m16n8k8.
//  8 warps (256 threads) in a 4×2 grid cover the 128×128 block tile.
//  Per k=8 step: 2 M-tiles × 8 N-tiles = 16 MMA calls per warp.
//  Per BK=16 tile: 32 MMA calls per warp.
//  Accumulators: 2×8×4 = 64 registers per thread.
//
//  Fragment mapping (mma.sync.m16n8k8.row.col.f32.tf32.tf32.f32):
//    lane = threadIdx.x % 32
//    gid  = lane / 4            (0..7)
//    tg   = lane % 4            (0..3)
//
//    A[m][k] row-major, stored transposed as As[k][m]:
//      a0 = As[kk + tg*2    ][m_base + gid    ]
//      a1 = As[kk + tg*2 + 1][m_base + gid    ]
//      a2 = As[kk + tg*2    ][m_base + gid + 8]
//      a3 = As[kk + tg*2 + 1][m_base + gid + 8]
//
//    B[k][n] col-major descriptor, stored as Bs[k][n]:
//      b0 = Bs[kk + tg*2    ][n_base + gid]
//      b1 = Bs[kk + tg*2 + 1][n_base + gid]
//
//    D[m][n] output:
//      d0 = C[m_base + gid    ][n_base + tg*2    ]
//      d1 = C[m_base + gid    ][n_base + tg*2 + 1]
//      d2 = C[m_base + gid + 8][n_base + tg*2    ]
//      d3 = C[m_base + gid + 8][n_base + tg*2 + 1]
// ===========================================================================

__global__ __launch_bounds__(NUM_THREADS, 2)
void sgemm_tc_kernel(const float* __restrict__ A,
                     const float* __restrict__ B,
                     float* __restrict__ C,
                     int M, int N, int K) {

    __shared__ float As[2][BK][SMEM_A_STRIDE];
    __shared__ float Bs[2][BK][SMEM_B_STRIDE];

    const int bm = blockIdx.y * BM;
    const int bn = blockIdx.x * BN;
    const int tid = threadIdx.x;

    // Warp / lane decomposition
    const int warp_id = tid / 32;
    const int lane = tid % 32;
    const int warp_m = warp_id / TC_WARPS_N;
    const int warp_n = warp_id % TC_WARPS_N;
    const int gid = lane / 4;
    const int tg  = lane % 4;

    // Accumulators
    float acc[MMA_TILES_M][MMA_TILES_N][4];
    #pragma unroll
    for (int mi = 0; mi < MMA_TILES_M; ++mi)
        #pragma unroll
        for (int ni = 0; ni < MMA_TILES_N; ++ni) {
            acc[mi][ni][0] = 0.f; acc[mi][ni][1] = 0.f;
            acc[mi][ni][2] = 0.f; acc[mi][ni][3] = 0.f;
        }

    // A load mapping — scalar, column-major for perfect global coalescing
    const int a_col = tid % BK;
    const int a_row0 = tid / BK;

    // B load mapping — 2 float4 per thread via cp.async
    const int b_row0 = tid / 32;
    const int b_col4 = tid % 32;
    const int b_row1 = 8 + b_row0;

    // ---- Tile loaders ----
    #define LOAD_A(buf, bk_off) do {                                         \
        const float* _Ab = A + bm * K + (bk_off) + a_col;                   \
        _Pragma("unroll")                                                     \
        for (int _s = 0; _s < 8; ++_s)                                      \
            As[buf][a_col][a_row0 + _s * 16] = _Ab[(a_row0 + _s * 16) * K]; \
    } while(0)

    #define LOAD_B(buf, bk_off) do {                                         \
        ptx::cp_async_cg_16(                                                  \
            &Bs[buf][b_row0][b_col4 * 4],                                    \
            &B[((bk_off) + b_row0) * N + bn + b_col4 * 4]);                 \
        ptx::cp_async_cg_16(                                                  \
            &Bs[buf][b_row1][b_col4 * 4],                                    \
            &B[((bk_off) + b_row1) * N + bn + b_col4 * 4]);                 \
    } while(0)

    // ---- MMA compute macro ----
    #define TC_COMPUTE(buf) do {                                             \
        _Pragma("unroll")                                                     \
        for (int _kk = 0; _kk < BK; _kk += MMA_K) {                        \
            _Pragma("unroll")                                                 \
            for (int _mi = 0; _mi < MMA_TILES_M; ++_mi) {                   \
                int _am = warp_m * TC_WM + _mi * MMA_M;                     \
                uint32_t _a0 = __float_as_uint(                              \
                    As[buf][_kk + tg*2    ][_am + gid    ]);                 \
                uint32_t _a1 = __float_as_uint(                              \
                    As[buf][_kk + tg*2    ][_am + gid + 8]);                \
                uint32_t _a2 = __float_as_uint(                              \
                    As[buf][_kk + tg*2 + 1][_am + gid    ]);                \
                uint32_t _a3 = __float_as_uint(                              \
                    As[buf][_kk + tg*2 + 1][_am + gid + 8]);                \
                _Pragma("unroll")                                             \
                for (int _ni = 0; _ni < MMA_TILES_N; ++_ni) {               \
                    int _bn = warp_n * TC_WN + _ni * MMA_N;                 \
                    uint32_t _b0 = __float_as_uint(                          \
                        Bs[buf][_kk + tg*2    ][_bn + gid]);                \
                    uint32_t _b1 = __float_as_uint(                          \
                        Bs[buf][_kk + tg*2 + 1][_bn + gid]);               \
                    ptx::mma_m16n8k8_tf32(                                   \
                        acc[_mi][_ni][0], acc[_mi][_ni][1],                  \
                        acc[_mi][_ni][2], acc[_mi][_ni][3],                  \
                        _a0, _a1, _a2, _a3, _b0, _b1,                       \
                        acc[_mi][_ni][0], acc[_mi][_ni][1],                  \
                        acc[_mi][_ni][2], acc[_mi][_ni][3]);                 \
                }                                                             \
            }                                                                 \
        }                                                                     \
    } while(0)

    // ---- Main loop with double buffering ----
    LOAD_A(0, 0);
    LOAD_B(0, 0);
    ptx::cp_async_commit();
    ptx::cp_async_wait_all();
    __syncthreads();

    const int num_tiles = K / BK;
    int buf = 0;

    for (int t = 0; t < num_tiles - 1; ++t) {
        LOAD_A(1 - buf, (t + 1) * BK);
        LOAD_B(1 - buf, (t + 1) * BK);
        ptx::cp_async_commit();

        TC_COMPUTE(buf);

        ptx::cp_async_wait_all();
        buf = 1 - buf;
        __syncthreads();
    }
    TC_COMPUTE(buf);

    #undef LOAD_A
    #undef LOAD_B
    #undef TC_COMPUTE

    // ---- Store C (float2 per output position) ----
    #pragma unroll
    for (int mi = 0; mi < MMA_TILES_M; ++mi) {
        int row0 = bm + warp_m * TC_WM + mi * MMA_M + gid;
        int row1 = row0 + 8;
        #pragma unroll
        for (int ni = 0; ni < MMA_TILES_N; ++ni) {
            int col = bn + warp_n * TC_WN + ni * MMA_N + tg * 2;
            *reinterpret_cast<float2*>(&C[row0 * N + col]) =
                make_float2(acc[mi][ni][0], acc[mi][ni][1]);
            *reinterpret_cast<float2*>(&C[row1 * N + col]) =
                make_float2(acc[mi][ni][2], acc[mi][ni][3]);
        }
    }
}

// ===========================================================================
//  Path 2: FP32 CUDA Core — fast path (aligned dims, no boundary checks)
// ===========================================================================

__global__ __launch_bounds__(NUM_THREADS, 2)
void sgemm_fast_kernel(const float* __restrict__ A,
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
    const int b_row0 = tid / 32;
    const int b_col4 = tid % 32;
    const int b_row1 = 8 + b_row0;

    float acc[TM][TN] = {};

    #define LOAD_A(buf, bk_off) do {                                         \
        const float* _Ab = A + bm * K + (bk_off) + a_col;                   \
        _Pragma("unroll")                                                     \
        for (int _s = 0; _s < 8; ++_s)                                      \
            As[buf][a_col][a_row0 + _s * 16] = _Ab[(a_row0 + _s * 16) * K]; \
    } while(0)

    #define LOAD_B_FAST(buf, bk_off) do {                                    \
        ptx::cp_async_cg_16(                                                  \
            &Bs[buf][b_row0][b_col4 * 4],                                    \
            &B[((bk_off) + b_row0) * N + bn + b_col4 * 4]);                 \
        ptx::cp_async_cg_16(                                                  \
            &Bs[buf][b_row1][b_col4 * 4],                                    \
            &B[((bk_off) + b_row1) * N + bn + b_col4 * 4]);                 \
    } while(0)

    #define COMPUTE(buf) do {                                                \
        _Pragma("unroll")                                                     \
        for (int _k = 0; _k < BK; ++_k) {                                   \
            float4 _a4lo = *reinterpret_cast<const float4*>(                 \
                               &As[buf][_k][a_base]);                        \
            float4 _a4hi = *reinterpret_cast<const float4*>(                 \
                               &As[buf][_k][a_base + 4]);                   \
            float4 _b4lo = *reinterpret_cast<const float4*>(                 \
                               &Bs[buf][_k][b_base]);                        \
            float4 _b4hi = *reinterpret_cast<const float4*>(                 \
                               &Bs[buf][_k][b_base + 4]);                   \
            float _af[TM] = {_a4lo.x, _a4lo.y, _a4lo.z, _a4lo.w,          \
                             _a4hi.x, _a4hi.y, _a4hi.z, _a4hi.w};          \
            float _bf[TN] = {_b4lo.x, _b4lo.y, _b4lo.z, _b4lo.w,          \
                             _b4hi.x, _b4hi.y, _b4hi.z, _b4hi.w};          \
            _Pragma("unroll")                                                 \
            for (int _m = 0; _m < TM; ++_m)                                 \
                _Pragma("unroll")                                             \
                for (int _n = 0; _n < TN; ++_n)                             \
                    acc[_m][_n] += _af[_m] * _bf[_n];                        \
        }                                                                     \
    } while(0)

    LOAD_A(0, 0);
    LOAD_B_FAST(0, 0);
    ptx::cp_async_commit();
    ptx::cp_async_wait_all();
    __syncthreads();

    const int num_tiles = K / BK;
    int buf = 0;

    for (int t = 0; t < num_tiles - 1; ++t) {
        LOAD_A(1 - buf, (t + 1) * BK);
        LOAD_B_FAST(1 - buf, (t + 1) * BK);
        ptx::cp_async_commit();
        COMPUTE(buf);
        ptx::cp_async_wait_all();
        buf = 1 - buf;
        __syncthreads();
    }
    COMPUTE(buf);

    #undef LOAD_A
    #undef LOAD_B_FAST
    #undef COMPUTE

    float* Crow = C + (bm + a_base) * N + bn + b_base;
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        *reinterpret_cast<float4*>(Crow)     = make_float4(acc[m][0], acc[m][1], acc[m][2], acc[m][3]);
        *reinterpret_cast<float4*>(Crow + 4) = make_float4(acc[m][4], acc[m][5], acc[m][6], acc[m][7]);
        Crow += N;
    }
}

// ===========================================================================
//  Path 3: FP32 CUDA Core — general (handles any M, N, K)
// ===========================================================================

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
    const int b_row0 = tid / 32;
    const int b_col4 = tid % 32;
    const int b_row1 = 8 + b_row0;

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
        auto do_one = [&](int br, int bc4) {
            int gr = bk + br;
            int gc = bn + bc4 * 4;
            if (gr < K && gc + 3 < N) {
                int addr = gr * N + gc;
                if ((addr & 3) == 0) {
                    *reinterpret_cast<float4*>(&Bs[buf_idx][br][bc4 * 4]) =
                        *reinterpret_cast<const float4*>(&B[addr]);
                } else {
                    Bs[buf_idx][br][bc4*4+0] = B[addr+0];
                    Bs[buf_idx][br][bc4*4+1] = B[addr+1];
                    Bs[buf_idx][br][bc4*4+2] = B[addr+2];
                    Bs[buf_idx][br][bc4*4+3] = B[addr+3];
                }
            } else {
                for (int j = 0; j < 4; ++j) {
                    int c = gc + j;
                    Bs[buf_idx][br][bc4*4+j] =
                        (gr < K && c < N) ? B[gr * N + c] : 0.0f;
                }
            }
        };
        do_one(b_row0, b_col4);
        do_one(b_row1, b_col4);
    };

    load_A_safe(0, 0);
    load_B_safe(0, 0);
    __syncthreads();

    int num_tiles = (K + BK - 1) / BK;
    int buf = 0;

    for (int t = 0; t < num_tiles - 1; ++t) {
        load_A_safe(1 - buf, (t + 1) * BK);
        load_B_safe(1 - buf, (t + 1) * BK);

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
//  Double-precision fallback (single-buffered, smaller tiles)
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
//  Launch dispatcher
// ===========================================================================

// Runtime TC check — the __CUDA_ARCH__ guard is for device code only,
// so we need a host-side capability check.
static bool has_tensor_cores() {
    static int cached = -1;
    if (cached < 0) {
        int major = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
        cached = (major >= 8) ? 1 : 0;
    }
    return cached == 1;
}

template <>
void launch_matmul_kernel<float>(const float* A, const float* B, float* C,
                                  int M, int N, int K, cudaStream_t stream) {
    dim3 blocks(ceil_div(N, BN), ceil_div(M, BM));
    bool aligned = (M % BM == 0) && (N % BN == 0) && (K % BK == 0);

    if (aligned && has_tensor_cores()) {
        sgemm_tc_kernel<<<blocks, NUM_THREADS, 0, stream>>>(A, B, C, M, N, K);
    } else if (aligned && (N % 4 == 0)) {
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
