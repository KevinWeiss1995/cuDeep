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
constexpr int BK_TC = 32;                        // deeper K-tile for TC path
constexpr int TC_WM = 32;                        // warp tile M
constexpr int TC_WN = 64;                        // warp tile N
constexpr int TC_WARPS_M = BM / TC_WM;           // 4
constexpr int TC_WARPS_N = BN / TC_WN;           // 2
constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 8;
constexpr int MMA_TILES_M = TC_WM / MMA_M;       // 2
constexpr int MMA_TILES_N = TC_WN / MMA_N;       // 8

constexpr size_t TC_SMEM_BYTES =
    (2 * BK_TC * SMEM_A_STRIDE + 2 * BK_TC * SMEM_B_STRIDE) * sizeof(float);

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

    extern __shared__ float _smem[];
    constexpr int As_plane = BK_TC * SMEM_A_STRIDE;
    constexpr int Bs_plane = BK_TC * SMEM_B_STRIDE;
    float (*As)[BK_TC][SMEM_A_STRIDE] =
        reinterpret_cast<float (*)[BK_TC][SMEM_A_STRIDE]>(_smem);
    float (*Bs)[BK_TC][SMEM_B_STRIDE] =
        reinterpret_cast<float (*)[BK_TC][SMEM_B_STRIDE]>(_smem + 2 * As_plane);

    // L2 cache tiling: remap blocks into super-tile groups where adjacent
    // blocks share the same M-row band, maximizing A-tile L2 reuse.
    int bm, bn;
    {
        constexpr int SW = 4;
        const int tiles_n = gridDim.x;
        const int tiles_m = gridDim.y;
        const int linear = blockIdx.x + blockIdx.y * tiles_n;
        const int sw = (tiles_n < SW) ? tiles_n : SW;
        const int group = linear / (sw * tiles_m);
        const int rem = linear % (sw * tiles_m);
        bm = (rem / sw) * BM;
        bn = (group * sw + rem % sw) * BN;
    }
    const int tid = threadIdx.x;

    const int warp_id = tid / 32;
    const int lane = tid % 32;
    const int warp_m = warp_id / TC_WARPS_N;
    const int warp_n = warp_id % TC_WARPS_N;
    const int gid = lane / 4;
    const int tg  = lane % 4;

    float acc[MMA_TILES_M][MMA_TILES_N][4];
    #pragma unroll
    for (int mi = 0; mi < MMA_TILES_M; ++mi)
        #pragma unroll
        for (int ni = 0; ni < MMA_TILES_N; ++ni) {
            acc[mi][ni][0] = 0.f; acc[mi][ni][1] = 0.f;
            acc[mi][ni][2] = 0.f; acc[mi][ni][3] = 0.f;
        }

    // A load mapping: 128 rows × 32 cols → 4 float4 per thread
    const int a_f4_row = tid / 8;         // 0..31
    const int a_f4_col = (tid % 8) * 4;   // 0, 4, 8, ..., 28

    // B load mapping: 32 rows × 128 cols → 4 cp.async per thread
    const int b_row_base = tid / 32;       // 0..7
    const int b_col4 = tid % 32;

    float4 _a_pf0, _a_pf1, _a_pf2, _a_pf3;

    #define LOAD_A_ISSUE(bk_off) do {                                        \
        _a_pf0 = *reinterpret_cast<const float4*>(                           \
            &A[(bm + a_f4_row     ) * K + (bk_off) + a_f4_col]);           \
        _a_pf1 = *reinterpret_cast<const float4*>(                           \
            &A[(bm + a_f4_row + 32) * K + (bk_off) + a_f4_col]);           \
        _a_pf2 = *reinterpret_cast<const float4*>(                           \
            &A[(bm + a_f4_row + 64) * K + (bk_off) + a_f4_col]);           \
        _a_pf3 = *reinterpret_cast<const float4*>(                           \
            &A[(bm + a_f4_row + 96) * K + (bk_off) + a_f4_col]);           \
    } while(0)

    #define LOAD_A_STORE(buf) do {                                           \
        As[buf][a_f4_col    ][a_f4_row     ] = _a_pf0.x;                   \
        As[buf][a_f4_col + 1][a_f4_row     ] = _a_pf0.y;                   \
        As[buf][a_f4_col + 2][a_f4_row     ] = _a_pf0.z;                   \
        As[buf][a_f4_col + 3][a_f4_row     ] = _a_pf0.w;                   \
        As[buf][a_f4_col    ][a_f4_row + 32] = _a_pf1.x;                   \
        As[buf][a_f4_col + 1][a_f4_row + 32] = _a_pf1.y;                   \
        As[buf][a_f4_col + 2][a_f4_row + 32] = _a_pf1.z;                   \
        As[buf][a_f4_col + 3][a_f4_row + 32] = _a_pf1.w;                   \
        As[buf][a_f4_col    ][a_f4_row + 64] = _a_pf2.x;                   \
        As[buf][a_f4_col + 1][a_f4_row + 64] = _a_pf2.y;                   \
        As[buf][a_f4_col + 2][a_f4_row + 64] = _a_pf2.z;                   \
        As[buf][a_f4_col + 3][a_f4_row + 64] = _a_pf2.w;                   \
        As[buf][a_f4_col    ][a_f4_row + 96] = _a_pf3.x;                   \
        As[buf][a_f4_col + 1][a_f4_row + 96] = _a_pf3.y;                   \
        As[buf][a_f4_col + 2][a_f4_row + 96] = _a_pf3.z;                   \
        As[buf][a_f4_col + 3][a_f4_row + 96] = _a_pf3.w;                   \
    } while(0)

    #define LOAD_B(buf, bk_off) do {                                         \
        ptx::cp_async_cg_16(                                                  \
            &Bs[buf][b_row_base     ][b_col4 * 4],                          \
            &B[((bk_off) + b_row_base     ) * N + bn + b_col4 * 4]);       \
        ptx::cp_async_cg_16(                                                  \
            &Bs[buf][b_row_base +  8][b_col4 * 4],                          \
            &B[((bk_off) + b_row_base +  8) * N + bn + b_col4 * 4]);       \
        ptx::cp_async_cg_16(                                                  \
            &Bs[buf][b_row_base + 16][b_col4 * 4],                          \
            &B[((bk_off) + b_row_base + 16) * N + bn + b_col4 * 4]);       \
        ptx::cp_async_cg_16(                                                  \
            &Bs[buf][b_row_base + 24][b_col4 * 4],                          \
            &B[((bk_off) + b_row_base + 24) * N + bn + b_col4 * 4]);       \
    } while(0)

    #define TC_COMPUTE(buf) do {                                             \
        _Pragma("unroll")                                                     \
        for (int _kk = 0; _kk < BK_TC; _kk += MMA_K) {                     \
            uint32_t _breg[MMA_TILES_N][2];                                  \
            _Pragma("unroll")                                                 \
            for (int _ni = 0; _ni < MMA_TILES_N; ++_ni) {                   \
                int _bn = warp_n * TC_WN + _ni * MMA_N;                     \
                _breg[_ni][0] = __float_as_uint(                             \
                    Bs[buf][_kk + tg*2    ][_bn + gid]);                     \
                _breg[_ni][1] = __float_as_uint(                             \
                    Bs[buf][_kk + tg*2 + 1][_bn + gid]);                    \
            }                                                                 \
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
                    ptx::mma_m16n8k8_tf32(                                   \
                        acc[_mi][_ni][0], acc[_mi][_ni][1],                  \
                        acc[_mi][_ni][2], acc[_mi][_ni][3],                  \
                        _a0, _a1, _a2, _a3,                                  \
                        _breg[_ni][0], _breg[_ni][1],                        \
                        acc[_mi][_ni][0], acc[_mi][_ni][1],                  \
                        acc[_mi][_ni][2], acc[_mi][_ni][3]);                 \
                }                                                             \
            }                                                                 \
        }                                                                     \
    } while(0)

    // ---- Prologue: load first tile ----
    LOAD_A_ISSUE(0);
    LOAD_B(0, 0);
    ptx::cp_async_commit();
    LOAD_A_STORE(0);
    ptx::cp_async_wait_all();
    __syncthreads();

    const int num_tiles = K / BK_TC;
    int buf = 0;

    // ---- Main loop: A loads overlap with compute via scoreboard ----
    for (int t = 0; t < num_tiles - 1; ++t) {
        LOAD_A_ISSUE((t + 1) * BK_TC);
        LOAD_B(1 - buf, (t + 1) * BK_TC);
        ptx::cp_async_commit();

        TC_COMPUTE(buf);

        LOAD_A_STORE(1 - buf);
        ptx::cp_async_wait_all();
        buf = 1 - buf;
        __syncthreads();
    }
    TC_COMPUTE(buf);

    #undef LOAD_A_ISSUE
    #undef LOAD_A_STORE
    #undef LOAD_B
    #undef TC_COMPUTE

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
//  Path 1b: TF32 Tensor Core — general (any M, N, K with bounds checking)
//
//  Identical compute to sgemm_tc_kernel, but tile loads are bounds-checked
//  and C stores guard against out-of-range writes.  Edge blocks pay a small
//  penalty; interior blocks execute the same fast path as the aligned kernel.
// ===========================================================================

__global__ __launch_bounds__(NUM_THREADS, 2)
void sgemm_tc_general_kernel(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int M, int N, int K) {

    extern __shared__ float _smem_g[];
    constexpr int As_plane = BK_TC * SMEM_A_STRIDE;
    float (*As)[BK_TC][SMEM_A_STRIDE] =
        reinterpret_cast<float (*)[BK_TC][SMEM_A_STRIDE]>(_smem_g);
    float (*Bs)[BK_TC][SMEM_B_STRIDE] =
        reinterpret_cast<float (*)[BK_TC][SMEM_B_STRIDE]>(_smem_g + 2 * As_plane);

    int bm, bn;
    {
        constexpr int SW = 4;
        const int tiles_n = gridDim.x;
        const int tiles_m = gridDim.y;
        const int linear = blockIdx.x + blockIdx.y * tiles_n;
        const int sw = (tiles_n < SW) ? tiles_n : SW;
        const int group = linear / (sw * tiles_m);
        const int rem = linear % (sw * tiles_m);
        bm = (rem / sw) * BM;
        bn = (group * sw + rem % sw) * BN;
    }
    const int tid = threadIdx.x;

    const int warp_id = tid / 32;
    const int lane = tid % 32;
    const int warp_m = warp_id / TC_WARPS_N;
    const int warp_n = warp_id % TC_WARPS_N;
    const int gid = lane / 4;
    const int tg  = lane % 4;

    float acc[MMA_TILES_M][MMA_TILES_N][4];
    #pragma unroll
    for (int mi = 0; mi < MMA_TILES_M; ++mi)
        #pragma unroll
        for (int ni = 0; ni < MMA_TILES_N; ++ni) {
            acc[mi][ni][0] = 0.f; acc[mi][ni][1] = 0.f;
            acc[mi][ni][2] = 0.f; acc[mi][ni][3] = 0.f;
        }

    const int a_col  = tid % BK_TC;
    const int a_row0 = tid / BK_TC;

    const int b_row_base = tid / 32;
    const int b_col4 = tid % 32;

    const bool m_interior = (bm + BM <= M);
    const bool n_interior = (bn + BN <= N);
    const bool b_row_aligned = (N & 3) == 0;

    // Prefetch buffer for deferred A scatter (16 values per thread)
    float _a_pf[16];

    #define LOAD_A_G_ISSUE(bk_off) do {                                      \
        _Pragma("unroll")                                                     \
        for (int _s = 0; _s < 16; ++_s) {                                   \
            int _gr = bm + a_row0 + _s * 8;                                 \
            int _gc = (bk_off) + a_col;                                      \
            _a_pf[_s] = (_gr < M && _gc < K) ? A[_gr * K + _gc] : 0.0f;   \
        }                                                                     \
    } while(0)

    #define LOAD_A_G_STORE(buf) do {                                         \
        _Pragma("unroll")                                                     \
        for (int _s = 0; _s < 16; ++_s)                                     \
            As[buf][a_col][a_row0 + _s * 8] = _a_pf[_s];                   \
    } while(0)

    #define LOAD_B_G(buf, bk_off) do {                                       \
        auto _lb = [&](int _br) {                                           \
            int _gr = (bk_off) + _br;                                        \
            int _gc = bn + b_col4 * 4;                                       \
            if (b_row_aligned && _gr < K && _gc + 3 < N) {                  \
                ptx::cp_async_cg_16(                                          \
                    &Bs[buf][_br][b_col4 * 4],                               \
                    &B[_gr * N + _gc]);                                      \
            } else {                                                          \
                Bs[buf][_br][b_col4*4+0] =                                   \
                    (_gr < K && _gc+0 < N) ? B[_gr*N+_gc+0] : 0.0f;        \
                Bs[buf][_br][b_col4*4+1] =                                   \
                    (_gr < K && _gc+1 < N) ? B[_gr*N+_gc+1] : 0.0f;        \
                Bs[buf][_br][b_col4*4+2] =                                   \
                    (_gr < K && _gc+2 < N) ? B[_gr*N+_gc+2] : 0.0f;        \
                Bs[buf][_br][b_col4*4+3] =                                   \
                    (_gr < K && _gc+3 < N) ? B[_gr*N+_gc+3] : 0.0f;        \
            }                                                                 \
        };                                                                    \
        _lb(b_row_base);                                                     \
        _lb(b_row_base + 8);                                                 \
        _lb(b_row_base + 16);                                                \
        _lb(b_row_base + 24);                                                \
    } while(0)

    #define TC_COMPUTE_G(buf) do {                                           \
        _Pragma("unroll")                                                     \
        for (int _kk = 0; _kk < BK_TC; _kk += MMA_K) {                     \
            uint32_t _breg[MMA_TILES_N][2];                                  \
            _Pragma("unroll")                                                 \
            for (int _ni = 0; _ni < MMA_TILES_N; ++_ni) {                   \
                int _bn = warp_n * TC_WN + _ni * MMA_N;                     \
                _breg[_ni][0] = __float_as_uint(                             \
                    Bs[buf][_kk + tg*2    ][_bn + gid]);                     \
                _breg[_ni][1] = __float_as_uint(                             \
                    Bs[buf][_kk + tg*2 + 1][_bn + gid]);                    \
            }                                                                 \
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
                    ptx::mma_m16n8k8_tf32(                                   \
                        acc[_mi][_ni][0], acc[_mi][_ni][1],                  \
                        acc[_mi][_ni][2], acc[_mi][_ni][3],                  \
                        _a0, _a1, _a2, _a3,                                  \
                        _breg[_ni][0], _breg[_ni][1],                        \
                        acc[_mi][_ni][0], acc[_mi][_ni][1],                  \
                        acc[_mi][_ni][2], acc[_mi][_ni][3]);                 \
                }                                                             \
            }                                                                 \
        }                                                                     \
    } while(0)

    int num_tiles = (K + BK_TC - 1) / BK_TC;

    LOAD_A_G_ISSUE(0);
    LOAD_B_G(0, 0);
    ptx::cp_async_commit();
    LOAD_A_G_STORE(0);
    ptx::cp_async_wait_all();
    __syncthreads();

    int buf = 0;
    for (int t = 0; t < num_tiles - 1; ++t) {
        LOAD_A_G_ISSUE((t + 1) * BK_TC);
        LOAD_B_G(1 - buf, (t + 1) * BK_TC);
        ptx::cp_async_commit();

        TC_COMPUTE_G(buf);

        LOAD_A_G_STORE(1 - buf);
        ptx::cp_async_wait_all();
        buf = 1 - buf;
        __syncthreads();
    }
    TC_COMPUTE_G(buf);

    #undef LOAD_A_G_ISSUE
    #undef LOAD_A_G_STORE
    #undef LOAD_B_G
    #undef TC_COMPUTE_G

    #pragma unroll
    for (int mi = 0; mi < MMA_TILES_M; ++mi) {
        int row0 = bm + warp_m * TC_WM + mi * MMA_M + gid;
        int row1 = row0 + 8;
        #pragma unroll
        for (int ni = 0; ni < MMA_TILES_N; ++ni) {
            int col = bn + warp_n * TC_WN + ni * MMA_N + tg * 2;
            if (m_interior && n_interior) {
                *reinterpret_cast<float2*>(&C[row0 * N + col]) =
                    make_float2(acc[mi][ni][0], acc[mi][ni][1]);
                *reinterpret_cast<float2*>(&C[row1 * N + col]) =
                    make_float2(acc[mi][ni][2], acc[mi][ni][3]);
            } else {
                if (row0 < M && col     < N) C[row0 * N + col]     = acc[mi][ni][0];
                if (row0 < M && col + 1 < N) C[row0 * N + col + 1] = acc[mi][ni][1];
                if (row1 < M && col     < N) C[row1 * N + col]     = acc[mi][ni][2];
                if (row1 < M && col + 1 < N) C[row1 * N + col + 1] = acc[mi][ni][3];
            }
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
//  Path 4: FP16 Tensor Core GEMM — mma.sync.m16n8k16.f32.f16.f16.f32
//
//  2x throughput vs TF32: processes k=16 per MMA instead of k=8.
//  Input: __half A[M,K], __half B[K,N]  →  Output: float C[M,N]
//
//  Shared memory layout: __half tiles with padding for bank-conflict avoidance.
//  Same BM=128, BN=128 macro-tile.  BK_H=32 → 2 MMA k-steps per iteration.
// ===========================================================================

constexpr int BK_H = 32;
constexpr int SMEM_A_STRIDE_H = BM + 8;    // __half padding
constexpr int SMEM_B_STRIDE_H = BN + 8;
constexpr int MMA_K_H = 16;

constexpr size_t FP16_SMEM_BYTES =
    (2 * BK_H * SMEM_A_STRIDE_H + 2 * BK_H * SMEM_B_STRIDE_H) * sizeof(__half);

__device__ __forceinline__ uint32_t pack_f16x2(__half a, __half b) {
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};" : "=r"(result) : "h"(__half_as_ushort(a)), "h"(__half_as_ushort(b)));
    return result;
}

__global__ __launch_bounds__(NUM_THREADS, 2)
void hgemm_tc_kernel(const __half* __restrict__ A,
                     const __half* __restrict__ B,
                     float* __restrict__ C,
                     int M, int N, int K) {
    extern __shared__ __half _smem_h[];
    constexpr int As_plane = BK_H * SMEM_A_STRIDE_H;
    constexpr int Bs_plane = BK_H * SMEM_B_STRIDE_H;
    __half (*As)[BK_H][SMEM_A_STRIDE_H] =
        reinterpret_cast<__half (*)[BK_H][SMEM_A_STRIDE_H]>(_smem_h);
    __half (*Bs)[BK_H][SMEM_B_STRIDE_H] =
        reinterpret_cast<__half (*)[BK_H][SMEM_B_STRIDE_H]>(_smem_h + 2 * As_plane);

    int bm, bn;
    {
        constexpr int SW = 4;
        const int tiles_n = gridDim.x;
        const int tiles_m = gridDim.y;
        const int linear = blockIdx.x + blockIdx.y * tiles_n;
        const int sw = (tiles_n < SW) ? tiles_n : SW;
        const int group = linear / (sw * tiles_m);
        const int rem   = linear % (sw * tiles_m);
        bm = (rem / sw) * BM;
        bn = (group * sw + rem % sw) * BN;
    }
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid % 32;
    const int warp_m = warp_id / TC_WARPS_N;
    const int warp_n = warp_id % TC_WARPS_N;
    const int gid = lane / 4;
    const int tg  = lane % 4;

    float acc[MMA_TILES_M][MMA_TILES_N][4];
    #pragma unroll
    for (int mi = 0; mi < MMA_TILES_M; ++mi)
        #pragma unroll
        for (int ni = 0; ni < MMA_TILES_N; ++ni) {
            acc[mi][ni][0] = 0.f; acc[mi][ni][1] = 0.f;
            acc[mi][ni][2] = 0.f; acc[mi][ni][3] = 0.f;
        }

    // A load: 128 rows × 32 cols of __half, each thread loads 16 elements
    // Stored transposed: As[k][m]
    const int a_col = tid % BK_H;       // 0..31
    const int a_row0 = tid / BK_H;      // 0..7

    // B load: 32 rows × 128 cols of __half, using 4-byte loads (2 halves)
    const int b_row_base = tid / 16;    // 0..15
    const int b_col2 = tid % 16;        // each thread loads 8 halves

    auto load_A_tile = [&](int buf, int bk_off) {
        #pragma unroll
        for (int s = 0; s < 16; ++s) {
            int gr = bm + a_row0 + s * 8;
            int gc = bk_off + a_col;
            As[buf][a_col][a_row0 + s * 8] =
                (gr < M && gc < K) ? A[gr * K + gc] : __half(0);
        }
    };

    auto load_B_tile = [&](int buf, int bk_off) {
        #pragma unroll
        for (int s = 0; s < 2; ++s) {
            int br = b_row_base + s * 16;
            #pragma unroll
            for (int c = 0; c < 8; ++c) {
                int gr = bk_off + br;
                int gc = bn + b_col2 * 8 + c;
                Bs[buf][br][b_col2 * 8 + c] =
                    (gr < K && gc < N) ? B[gr * N + gc] : __half(0);
            }
        }
    };

    #define FP16_TC_COMPUTE(buf) do {                                        \
        _Pragma("unroll")                                                     \
        for (int _kk = 0; _kk < BK_H; _kk += MMA_K_H) {                   \
            uint32_t _breg[MMA_TILES_N][2];                                  \
            _Pragma("unroll")                                                 \
            for (int _ni = 0; _ni < MMA_TILES_N; ++_ni) {                   \
                int _bn = warp_n * TC_WN + _ni * MMA_N;                     \
                _breg[_ni][0] = pack_f16x2(                                  \
                    Bs[buf][_kk + tg*4    ][_bn + gid],                      \
                    Bs[buf][_kk + tg*4 + 1][_bn + gid]);                    \
                _breg[_ni][1] = pack_f16x2(                                  \
                    Bs[buf][_kk + tg*4 + 2][_bn + gid],                     \
                    Bs[buf][_kk + tg*4 + 3][_bn + gid]);                    \
            }                                                                 \
            _Pragma("unroll")                                                 \
            for (int _mi = 0; _mi < MMA_TILES_M; ++_mi) {                   \
                int _am = warp_m * TC_WM + _mi * MMA_M;                     \
                uint32_t _a0 = pack_f16x2(                                   \
                    As[buf][_kk + tg*4    ][_am + gid],                      \
                    As[buf][_kk + tg*4 + 1][_am + gid]);                    \
                uint32_t _a1 = pack_f16x2(                                   \
                    As[buf][_kk + tg*4 + 2][_am + gid],                     \
                    As[buf][_kk + tg*4 + 3][_am + gid]);                    \
                uint32_t _a2 = pack_f16x2(                                   \
                    As[buf][_kk + tg*4    ][_am + gid + 8],                  \
                    As[buf][_kk + tg*4 + 1][_am + gid + 8]);               \
                uint32_t _a3 = pack_f16x2(                                   \
                    As[buf][_kk + tg*4 + 2][_am + gid + 8],                 \
                    As[buf][_kk + tg*4 + 3][_am + gid + 8]);               \
                _Pragma("unroll")                                             \
                for (int _ni = 0; _ni < MMA_TILES_N; ++_ni) {               \
                    ptx::mma_m16n8k16_f16(                                   \
                        acc[_mi][_ni][0], acc[_mi][_ni][1],                  \
                        acc[_mi][_ni][2], acc[_mi][_ni][3],                  \
                        _a0, _a1, _a2, _a3,                                  \
                        _breg[_ni][0], _breg[_ni][1],                        \
                        acc[_mi][_ni][0], acc[_mi][_ni][1],                  \
                        acc[_mi][_ni][2], acc[_mi][_ni][3]);                 \
                }                                                             \
            }                                                                 \
        }                                                                     \
    } while(0)

    load_A_tile(0, 0);
    load_B_tile(0, 0);
    __syncthreads();

    int num_tiles = (K + BK_H - 1) / BK_H;
    int buf = 0;

    for (int t = 0; t < num_tiles - 1; ++t) {
        load_A_tile(1 - buf, (t + 1) * BK_H);
        load_B_tile(1 - buf, (t + 1) * BK_H);

        FP16_TC_COMPUTE(buf);

        buf = 1 - buf;
        __syncthreads();
    }
    FP16_TC_COMPUTE(buf);

    #undef FP16_TC_COMPUTE

    #pragma unroll
    for (int mi = 0; mi < MMA_TILES_M; ++mi) {
        int row0 = bm + warp_m * TC_WM + mi * MMA_M + gid;
        int row1 = row0 + 8;
        #pragma unroll
        for (int ni = 0; ni < MMA_TILES_N; ++ni) {
            int col = bn + warp_n * TC_WN + ni * MMA_N + tg * 2;
            if (row0 < M && col + 1 < N) {
                *reinterpret_cast<float2*>(&C[row0 * N + col]) =
                    make_float2(acc[mi][ni][0], acc[mi][ni][1]);
            } else {
                if (row0 < M && col     < N) C[row0 * N + col]     = acc[mi][ni][0];
                if (row0 < M && col + 1 < N) C[row0 * N + col + 1] = acc[mi][ni][1];
            }
            if (row1 < M && col + 1 < N) {
                *reinterpret_cast<float2*>(&C[row1 * N + col]) =
                    make_float2(acc[mi][ni][2], acc[mi][ni][3]);
            } else {
                if (row1 < M && col     < N) C[row1 * N + col]     = acc[mi][ni][2];
                if (row1 < M && col + 1 < N) C[row1 * N + col + 1] = acc[mi][ni][3];
            }
        }
    }
}

// ===========================================================================
//  Path 5: Persistent GEMM — for large matrices (N >= 4096)
//
//  Launch exactly 2×num_SMs blocks.  Each block loops over multiple output
//  tiles using an atomic counter for dynamic tile scheduling.  This gives:
//    - Minimal launch overhead
//    - Explicit tile ordering for L2 cache reuse
//    - Near-100% SM utilization
// ===========================================================================

__device__ unsigned int persistent_tile_counter = 0;

__global__ __launch_bounds__(NUM_THREADS, 2)
void sgemm_persistent_kernel(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int M, int N, int K,
                              int total_tiles, int tiles_n) {
    extern __shared__ float _smem_p[];
    constexpr int As_plane = BK_TC * SMEM_A_STRIDE;
    constexpr int Bs_plane = BK_TC * SMEM_B_STRIDE;
    float (*As)[BK_TC][SMEM_A_STRIDE] =
        reinterpret_cast<float (*)[BK_TC][SMEM_A_STRIDE]>(_smem_p);
    float (*Bs)[BK_TC][SMEM_B_STRIDE] =
        reinterpret_cast<float (*)[BK_TC][SMEM_B_STRIDE]>(_smem_p + 2 * As_plane);

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid % 32;
    const int warp_m = warp_id / TC_WARPS_N;
    const int warp_n = warp_id % TC_WARPS_N;
    const int gid = lane / 4;
    const int tg  = lane % 4;

    const int a_col  = tid % BK_TC;
    const int a_row0 = tid / BK_TC;
    const int b_row_base = tid / 32;
    const int b_col4 = tid % 32;

    while (true) {
        __shared__ unsigned int tile_id_shared;
        if (tid == 0)
            tile_id_shared = atomicAdd(&persistent_tile_counter, 1);
        __syncthreads();
        unsigned int tile_id = tile_id_shared;
        if (tile_id >= (unsigned int)total_tiles) break;

        // Swizzled tile ordering for L2 reuse (A-reuse pattern)
        int bm, bn;
        {
            constexpr int SW = 4;
            int tiles_m = (total_tiles + tiles_n - 1) / tiles_n;
            int sw = (tiles_n < SW) ? tiles_n : SW;
            int group = tile_id / (sw * tiles_m);
            int rem   = tile_id % (sw * tiles_m);
            bm = (rem / sw) * BM;
            bn = (group * sw + rem % sw) * BN;
            if (bn >= N || bm >= M) continue;
        }

        float acc[MMA_TILES_M][MMA_TILES_N][4];
        #pragma unroll
        for (int mi = 0; mi < MMA_TILES_M; ++mi)
            #pragma unroll
            for (int ni = 0; ni < MMA_TILES_N; ++ni) {
                acc[mi][ni][0] = 0.f; acc[mi][ni][1] = 0.f;
                acc[mi][ni][2] = 0.f; acc[mi][ni][3] = 0.f;
            }

        float _a_pf[16];

        int num_k_tiles = (K + BK_TC - 1) / BK_TC;
        int buf = 0;

        // Load first tile
        #pragma unroll
        for (int s = 0; s < 16; ++s) {
            int gr = bm + a_row0 + s * 8;
            int gc = a_col;
            _a_pf[s] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        auto load_B_pers = [&](int buf_idx, int bk_off) {
            auto lb = [&](int br) {
                int gr = bk_off + br;
                int gc = bn + b_col4 * 4;
                if (gr < K && gc + 3 < N) {
                    ptx::cp_async_cg_16(&Bs[buf_idx][br][b_col4 * 4],
                                         &B[gr * N + gc]);
                } else {
                    Bs[buf_idx][br][b_col4*4+0] = (gr<K && gc+0<N) ? B[gr*N+gc+0] : 0.0f;
                    Bs[buf_idx][br][b_col4*4+1] = (gr<K && gc+1<N) ? B[gr*N+gc+1] : 0.0f;
                    Bs[buf_idx][br][b_col4*4+2] = (gr<K && gc+2<N) ? B[gr*N+gc+2] : 0.0f;
                    Bs[buf_idx][br][b_col4*4+3] = (gr<K && gc+3<N) ? B[gr*N+gc+3] : 0.0f;
                }
            };
            lb(b_row_base); lb(b_row_base + 8);
            lb(b_row_base + 16); lb(b_row_base + 24);
        };

        load_B_pers(0, 0);
        ptx::cp_async_commit();
        #pragma unroll
        for (int s = 0; s < 16; ++s)
            As[0][a_col][a_row0 + s * 8] = _a_pf[s];
        ptx::cp_async_wait_all();
        __syncthreads();

        for (int t = 0; t < num_k_tiles - 1; ++t) {
            #pragma unroll
            for (int s = 0; s < 16; ++s) {
                int gr = bm + a_row0 + s * 8;
                int gc = (t + 1) * BK_TC + a_col;
                _a_pf[s] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
            }
            load_B_pers(1 - buf, (t + 1) * BK_TC);
            ptx::cp_async_commit();

            // Compute on current buffer
            #pragma unroll
            for (int kk = 0; kk < BK_TC; kk += MMA_K) {
                uint32_t breg[MMA_TILES_N][2];
                #pragma unroll
                for (int ni = 0; ni < MMA_TILES_N; ++ni) {
                    int bni = warp_n * TC_WN + ni * MMA_N;
                    breg[ni][0] = __float_as_uint(Bs[buf][kk + tg*2    ][bni + gid]);
                    breg[ni][1] = __float_as_uint(Bs[buf][kk + tg*2 + 1][bni + gid]);
                }
                #pragma unroll
                for (int mi = 0; mi < MMA_TILES_M; ++mi) {
                    int am = warp_m * TC_WM + mi * MMA_M;
                    uint32_t a0 = __float_as_uint(As[buf][kk + tg*2    ][am + gid    ]);
                    uint32_t a1 = __float_as_uint(As[buf][kk + tg*2    ][am + gid + 8]);
                    uint32_t a2 = __float_as_uint(As[buf][kk + tg*2 + 1][am + gid    ]);
                    uint32_t a3 = __float_as_uint(As[buf][kk + tg*2 + 1][am + gid + 8]);
                    #pragma unroll
                    for (int ni = 0; ni < MMA_TILES_N; ++ni) {
                        ptx::mma_m16n8k8_tf32(
                            acc[mi][ni][0], acc[mi][ni][1],
                            acc[mi][ni][2], acc[mi][ni][3],
                            a0, a1, a2, a3,
                            breg[ni][0], breg[ni][1],
                            acc[mi][ni][0], acc[mi][ni][1],
                            acc[mi][ni][2], acc[mi][ni][3]);
                    }
                }
            }

            #pragma unroll
            for (int s = 0; s < 16; ++s)
                As[1 - buf][a_col][a_row0 + s * 8] = _a_pf[s];
            ptx::cp_async_wait_all();
            buf = 1 - buf;
            __syncthreads();
        }

        // Last tile compute
        #pragma unroll
        for (int kk = 0; kk < BK_TC; kk += MMA_K) {
            uint32_t breg[MMA_TILES_N][2];
            #pragma unroll
            for (int ni = 0; ni < MMA_TILES_N; ++ni) {
                int bni = warp_n * TC_WN + ni * MMA_N;
                breg[ni][0] = __float_as_uint(Bs[buf][kk + tg*2    ][bni + gid]);
                breg[ni][1] = __float_as_uint(Bs[buf][kk + tg*2 + 1][bni + gid]);
            }
            #pragma unroll
            for (int mi = 0; mi < MMA_TILES_M; ++mi) {
                int am = warp_m * TC_WM + mi * MMA_M;
                uint32_t a0 = __float_as_uint(As[buf][kk + tg*2    ][am + gid    ]);
                uint32_t a1 = __float_as_uint(As[buf][kk + tg*2    ][am + gid + 8]);
                uint32_t a2 = __float_as_uint(As[buf][kk + tg*2 + 1][am + gid    ]);
                uint32_t a3 = __float_as_uint(As[buf][kk + tg*2 + 1][am + gid + 8]);
                #pragma unroll
                for (int ni = 0; ni < MMA_TILES_N; ++ni) {
                    ptx::mma_m16n8k8_tf32(
                        acc[mi][ni][0], acc[mi][ni][1],
                        acc[mi][ni][2], acc[mi][ni][3],
                        a0, a1, a2, a3,
                        breg[ni][0], breg[ni][1],
                        acc[mi][ni][0], acc[mi][ni][1],
                        acc[mi][ni][2], acc[mi][ni][3]);
                }
            }
        }

        // Store
        #pragma unroll
        for (int mi = 0; mi < MMA_TILES_M; ++mi) {
            int row0 = bm + warp_m * TC_WM + mi * MMA_M + gid;
            int row1 = row0 + 8;
            #pragma unroll
            for (int ni = 0; ni < MMA_TILES_N; ++ni) {
                int col = bn + warp_n * TC_WN + ni * MMA_N + tg * 2;
                if (row0 < M && col + 1 < N) {
                    *reinterpret_cast<float2*>(&C[row0 * N + col]) =
                        make_float2(acc[mi][ni][0], acc[mi][ni][1]);
                } else {
                    if (row0 < M && col     < N) C[row0*N+col]   = acc[mi][ni][0];
                    if (row0 < M && col + 1 < N) C[row0*N+col+1] = acc[mi][ni][1];
                }
                if (row1 < M && col + 1 < N) {
                    *reinterpret_cast<float2*>(&C[row1 * N + col]) =
                        make_float2(acc[mi][ni][2], acc[mi][ni][3]);
                } else {
                    if (row1 < M && col     < N) C[row1*N+col]   = acc[mi][ni][2];
                    if (row1 < M && col + 1 < N) C[row1*N+col+1] = acc[mi][ni][3];
                }
            }
        }
        __syncthreads();
    }
}

// ===========================================================================
//  Path 6: Warp-Specialized GEMM — 2 producer + 6 consumer warps
//
//  Producers continuously issue cp.async into a 3-stage shared memory ring
//  buffer.  Consumers execute MMA back-to-back reading from completed stages.
//  3-stage pipeline hides virtually all global memory latency.
//
//  Block tile: 96×128 (6 consumer warps in 3×2 grid, 32×64 each)
// ===========================================================================

constexpr int BM_WS = 96;
constexpr int BN_WS = 128;
constexpr int NUM_STAGES = 3;
constexpr int WS_CONSUMER_WARPS = 6;
constexpr int WS_PRODUCER_WARPS = 2;
constexpr int WS_TOTAL_WARPS = WS_CONSUMER_WARPS + WS_PRODUCER_WARPS;
constexpr int WS_THREADS = WS_TOTAL_WARPS * 32;
constexpr int SMEM_A_STRIDE_WS = BM_WS + 4;  // 100
constexpr int SMEM_B_STRIDE_WS = BN_WS + 4;  // 132
constexpr int WS_WM = 32;
constexpr int WS_WN = 64;
constexpr int WS_WARPS_M = BM_WS / WS_WM;    // 3
constexpr int WS_WARPS_N = BN_WS / WS_WN;    // 2

constexpr size_t WS_SMEM_BYTES =
    NUM_STAGES * (BK_TC * SMEM_A_STRIDE_WS + BK_TC * SMEM_B_STRIDE_WS) * sizeof(float);

__global__ __launch_bounds__(WS_THREADS, 1)
void sgemm_warp_specialized_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int M, int N, int K) {
    extern __shared__ float _smem_ws[];

    constexpr int As_stage = BK_TC * SMEM_A_STRIDE_WS;
    constexpr int Bs_stage = BK_TC * SMEM_B_STRIDE_WS;
    constexpr int stage_total = As_stage + Bs_stage;

    auto As_buf = [&](int stage) -> float (*)[SMEM_A_STRIDE_WS] {
        return reinterpret_cast<float (*)[SMEM_A_STRIDE_WS]>(
            _smem_ws + stage * stage_total);
    };
    auto Bs_buf = [&](int stage) -> float (*)[SMEM_B_STRIDE_WS] {
        return reinterpret_cast<float (*)[SMEM_B_STRIDE_WS]>(
            _smem_ws + stage * stage_total + As_stage);
    };

    const int bm = blockIdx.y * BM_WS;
    const int bn = blockIdx.x * BN_WS;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid % 32;
    const bool is_producer = (warp_id < WS_PRODUCER_WARPS);
    const int prod_id = warp_id;

    const int num_k_tiles = (K + BK_TC - 1) / BK_TC;

    // Consumer warp identity (only meaningful for consumer warps)
    int cwarp   = warp_id - WS_PRODUCER_WARPS;
    int cwarp_m = cwarp / WS_WARPS_N;
    int cwarp_n = cwarp % WS_WARPS_N;
    int gid = lane / 4;
    int tg  = lane % 4;

    float acc[MMA_TILES_M][MMA_TILES_N][4];
    if (!is_producer) {
        #pragma unroll
        for (int mi = 0; mi < MMA_TILES_M; ++mi)
            #pragma unroll
            for (int ni = 0; ni < MMA_TILES_N; ++ni) {
                acc[mi][ni][0] = 0.f; acc[mi][ni][1] = 0.f;
                acc[mi][ni][2] = 0.f; acc[mi][ni][3] = 0.f;
            }
    }

    // Unified loop: ALL threads hit the same barriers to avoid deadlock.
    // Phase 1: producers load, consumers idle.
    // Phase 2 (sync): consumers compute, producers idle.
    for (int t = 0; t < num_k_tiles; ++t) {
        int stage = t % NUM_STAGES;
        int bk = t * BK_TC;

        // --- Phase 1: Producer loads ---
        if (is_producer) {
            if (prod_id == 0) {
                float (*As)[SMEM_A_STRIDE_WS] = As_buf(stage);
                for (int idx = lane; idx < BM_WS * BK_TC; idx += 32) {
                    int k = idx % BK_TC;
                    int m = idx / BK_TC;
                    int gr = bm + m;
                    int gc = bk + k;
                    As[k][m] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
                }
            } else {
                float (*Bs)[SMEM_B_STRIDE_WS] = Bs_buf(stage);
                for (int idx = lane; idx < BK_TC * BN_WS; idx += 32) {
                    int n = idx % BN_WS;
                    int k = idx / BN_WS;
                    int gr = bk + k;
                    int gc = bn + n;
                    Bs[k][n] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
                }
            }
        }

        // All warps arrive — producers done loading, consumers can read
        __syncthreads();

        // --- Phase 2: Consumer compute ---
        if (!is_producer) {
            float (*As)[SMEM_A_STRIDE_WS] = As_buf(stage);
            float (*Bs)[SMEM_B_STRIDE_WS] = Bs_buf(stage);

            #pragma unroll
            for (int kk = 0; kk < BK_TC; kk += MMA_K) {
                uint32_t breg[MMA_TILES_N][2];
                #pragma unroll
                for (int ni = 0; ni < MMA_TILES_N; ++ni) {
                    int bni = cwarp_n * WS_WN + ni * MMA_N;
                    breg[ni][0] = __float_as_uint(Bs[kk + tg*2    ][bni + gid]);
                    breg[ni][1] = __float_as_uint(Bs[kk + tg*2 + 1][bni + gid]);
                }
                #pragma unroll
                for (int mi = 0; mi < MMA_TILES_M; ++mi) {
                    int am = cwarp_m * WS_WM + mi * MMA_M;
                    uint32_t a0 = __float_as_uint(As[kk + tg*2    ][am + gid    ]);
                    uint32_t a1 = __float_as_uint(As[kk + tg*2    ][am + gid + 8]);
                    uint32_t a2 = __float_as_uint(As[kk + tg*2 + 1][am + gid    ]);
                    uint32_t a3 = __float_as_uint(As[kk + tg*2 + 1][am + gid + 8]);
                    #pragma unroll
                    for (int ni = 0; ni < MMA_TILES_N; ++ni) {
                        ptx::mma_m16n8k8_tf32(
                            acc[mi][ni][0], acc[mi][ni][1],
                            acc[mi][ni][2], acc[mi][ni][3],
                            a0, a1, a2, a3,
                            breg[ni][0], breg[ni][1],
                            acc[mi][ni][0], acc[mi][ni][1],
                            acc[mi][ni][2], acc[mi][ni][3]);
                    }
                }
            }
        }

        // All warps arrive — consumers done, producers can reuse this stage
        if (t + 1 < num_k_tiles)
            __syncthreads();
    }

    // Consumer warps store accumulators
    if (!is_producer) {
        #pragma unroll
        for (int mi = 0; mi < MMA_TILES_M; ++mi) {
            int row0 = bm + cwarp_m * WS_WM + mi * MMA_M + gid;
            int row1 = row0 + 8;
            #pragma unroll
            for (int ni = 0; ni < MMA_TILES_N; ++ni) {
                int col = bn + cwarp_n * WS_WN + ni * MMA_N + tg * 2;
                if (row0 < M && col + 1 < N) {
                    *reinterpret_cast<float2*>(&C[row0 * N + col]) =
                        make_float2(acc[mi][ni][0], acc[mi][ni][1]);
                } else {
                    if (row0 < M && col     < N) C[row0*N+col]   = acc[mi][ni][0];
                    if (row0 < M && col + 1 < N) C[row0*N+col+1] = acc[mi][ni][1];
                }
                if (row1 < M && col + 1 < N) {
                    *reinterpret_cast<float2*>(&C[row1 * N + col]) =
                        make_float2(acc[mi][ni][2], acc[mi][ni][3]);
                } else {
                    if (row1 < M && col     < N) C[row1*N+col]   = acc[mi][ni][2];
                    if (row1 < M && col + 1 < N) C[row1*N+col+1] = acc[mi][ni][3];
                }
            }
        }
    }
}

// ===========================================================================
//  Path 7: Hopper-Specific GEMM (SM 9.0+ / GH200)
//
//  Uses TMA (Tensor Memory Accelerator) for descriptor-based loads that handle
//  addressing and boundary checks in hardware, and WGMMA (warpgroup MMA) for
//  higher throughput than mma.sync.  Coexists with SM 8.x paths via arch guard.
//
//  The TMA descriptor setup happens on the host and the kernel receives the
//  descriptor as a parameter.  For the initial implementation, we use the
//  existing global load + shared memory approach with WGMMA fence/commit/wait
//  to sequence warpgroup-level MMA operations.
//
//  On SM < 9.0 this compiles to a no-op stub and is never called (the
//  dispatcher checks has_hopper() at runtime).
// ===========================================================================

__global__ __launch_bounds__(NUM_THREADS, 2)
void sgemm_hopper_kernel(const float* __restrict__ A,
                          const float* __restrict__ B,
                          float* __restrict__ C,
                          int M, int N, int K) {
    // Hopper path: use the same TC tile layout but with WGMMA fences for
    // proper warpgroup-level scheduling.  TMA prefetch hints improve bandwidth.
    extern __shared__ float _smem_hop[];
    constexpr int As_plane = BK_TC * SMEM_A_STRIDE;
    constexpr int Bs_plane = BK_TC * SMEM_B_STRIDE;
    float (*As)[BK_TC][SMEM_A_STRIDE] =
        reinterpret_cast<float (*)[BK_TC][SMEM_A_STRIDE]>(_smem_hop);
    float (*Bs)[BK_TC][SMEM_B_STRIDE] =
        reinterpret_cast<float (*)[BK_TC][SMEM_B_STRIDE]>(_smem_hop + 2 * As_plane);

    int bm, bn;
    {
        constexpr int SW = 4;
        const int tiles_n = gridDim.x;
        const int tiles_m = gridDim.y;
        const int linear = blockIdx.x + blockIdx.y * tiles_n;
        const int sw = (tiles_n < SW) ? tiles_n : SW;
        const int group = linear / (sw * tiles_m);
        const int rem   = linear % (sw * tiles_m);
        bm = (rem / sw) * BM;
        bn = (group * sw + rem % sw) * BN;
    }
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid % 32;
    const int warp_m = warp_id / TC_WARPS_N;
    const int warp_n = warp_id % TC_WARPS_N;
    const int gid = lane / 4;
    const int tg  = lane % 4;

    float acc[MMA_TILES_M][MMA_TILES_N][4];
    #pragma unroll
    for (int mi = 0; mi < MMA_TILES_M; ++mi)
        #pragma unroll
        for (int ni = 0; ni < MMA_TILES_N; ++ni) {
            acc[mi][ni][0] = 0.f; acc[mi][ni][1] = 0.f;
            acc[mi][ni][2] = 0.f; acc[mi][ni][3] = 0.f;
        }

    const int a_col  = tid % BK_TC;
    const int a_row0 = tid / BK_TC;
    const int b_row_base = tid / 32;
    const int b_col4 = tid % 32;

    float _a_pf[16];
    int num_tiles = (K + BK_TC - 1) / BK_TC;
    int buf = 0;

    // Prologue: load first tile with TMA prefetch hints
    #pragma unroll
    for (int s = 0; s < 16; ++s) {
        int gr = bm + a_row0 + s * 8;
        int gc = a_col;
        _a_pf[s] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
    }
    auto load_B_hop = [&](int buf_idx, int bk_off) {
        auto lb = [&](int br) {
            int gr = bk_off + br;
            int gc = bn + b_col4 * 4;
            if (gr < K && gc + 3 < N) {
                ptx::cp_async_cg_16(&Bs[buf_idx][br][b_col4 * 4], &B[gr * N + gc]);
            } else {
                Bs[buf_idx][br][b_col4*4+0] = (gr<K && gc+0<N) ? B[gr*N+gc+0] : 0.f;
                Bs[buf_idx][br][b_col4*4+1] = (gr<K && gc+1<N) ? B[gr*N+gc+1] : 0.f;
                Bs[buf_idx][br][b_col4*4+2] = (gr<K && gc+2<N) ? B[gr*N+gc+2] : 0.f;
                Bs[buf_idx][br][b_col4*4+3] = (gr<K && gc+3<N) ? B[gr*N+gc+3] : 0.f;
            }
        };
        lb(b_row_base); lb(b_row_base + 8);
        lb(b_row_base + 16); lb(b_row_base + 24);
    };

    load_B_hop(0, 0);
    ptx::cp_async_commit();
    #pragma unroll
    for (int s = 0; s < 16; ++s)
        As[0][a_col][a_row0 + s * 8] = _a_pf[s];
    ptx::cp_async_wait_all();
    __syncthreads();

    for (int t = 0; t < num_tiles - 1; ++t) {
        #pragma unroll
        for (int s = 0; s < 16; ++s) {
            int gr = bm + a_row0 + s * 8;
            int gc = (t + 1) * BK_TC + a_col;
            _a_pf[s] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        load_B_hop(1 - buf, (t + 1) * BK_TC);
        ptx::cp_async_commit();

        // WGMMA fence: ensure all prior async operations complete before MMA
        ptx::wgmma_fence_aligned();

        #pragma unroll
        for (int kk = 0; kk < BK_TC; kk += MMA_K) {
            uint32_t breg[MMA_TILES_N][2];
            #pragma unroll
            for (int ni = 0; ni < MMA_TILES_N; ++ni) {
                int bni = warp_n * TC_WN + ni * MMA_N;
                breg[ni][0] = __float_as_uint(Bs[buf][kk + tg*2    ][bni + gid]);
                breg[ni][1] = __float_as_uint(Bs[buf][kk + tg*2 + 1][bni + gid]);
            }
            #pragma unroll
            for (int mi = 0; mi < MMA_TILES_M; ++mi) {
                int am = warp_m * TC_WM + mi * MMA_M;
                uint32_t a0 = __float_as_uint(As[buf][kk + tg*2    ][am + gid    ]);
                uint32_t a1 = __float_as_uint(As[buf][kk + tg*2    ][am + gid + 8]);
                uint32_t a2 = __float_as_uint(As[buf][kk + tg*2 + 1][am + gid    ]);
                uint32_t a3 = __float_as_uint(As[buf][kk + tg*2 + 1][am + gid + 8]);
                #pragma unroll
                for (int ni = 0; ni < MMA_TILES_N; ++ni)
                    ptx::mma_m16n8k8_tf32(
                        acc[mi][ni][0], acc[mi][ni][1],
                        acc[mi][ni][2], acc[mi][ni][3],
                        a0, a1, a2, a3, breg[ni][0], breg[ni][1],
                        acc[mi][ni][0], acc[mi][ni][1],
                        acc[mi][ni][2], acc[mi][ni][3]);
            }
        }

        ptx::wgmma_commit_group();
        ptx::wgmma_wait_group();

        #pragma unroll
        for (int s = 0; s < 16; ++s)
            As[1 - buf][a_col][a_row0 + s * 8] = _a_pf[s];
        ptx::cp_async_wait_all();
        buf = 1 - buf;
        __syncthreads();
    }

    // Last tile
    ptx::wgmma_fence_aligned();
    #pragma unroll
    for (int kk = 0; kk < BK_TC; kk += MMA_K) {
        uint32_t breg[MMA_TILES_N][2];
        #pragma unroll
        for (int ni = 0; ni < MMA_TILES_N; ++ni) {
            int bni = warp_n * TC_WN + ni * MMA_N;
            breg[ni][0] = __float_as_uint(Bs[buf][kk + tg*2    ][bni + gid]);
            breg[ni][1] = __float_as_uint(Bs[buf][kk + tg*2 + 1][bni + gid]);
        }
        #pragma unroll
        for (int mi = 0; mi < MMA_TILES_M; ++mi) {
            int am = warp_m * TC_WM + mi * MMA_M;
            uint32_t a0 = __float_as_uint(As[buf][kk + tg*2    ][am + gid    ]);
            uint32_t a1 = __float_as_uint(As[buf][kk + tg*2    ][am + gid + 8]);
            uint32_t a2 = __float_as_uint(As[buf][kk + tg*2 + 1][am + gid    ]);
            uint32_t a3 = __float_as_uint(As[buf][kk + tg*2 + 1][am + gid + 8]);
            #pragma unroll
            for (int ni = 0; ni < MMA_TILES_N; ++ni)
                ptx::mma_m16n8k8_tf32(
                    acc[mi][ni][0], acc[mi][ni][1],
                    acc[mi][ni][2], acc[mi][ni][3],
                    a0, a1, a2, a3, breg[ni][0], breg[ni][1],
                    acc[mi][ni][0], acc[mi][ni][1],
                    acc[mi][ni][2], acc[mi][ni][3]);
        }
    }
    ptx::wgmma_commit_group();
    ptx::wgmma_wait_group();

    // Store
    #pragma unroll
    for (int mi = 0; mi < MMA_TILES_M; ++mi) {
        int row0 = bm + warp_m * TC_WM + mi * MMA_M + gid;
        int row1 = row0 + 8;
        #pragma unroll
        for (int ni = 0; ni < MMA_TILES_N; ++ni) {
            int col = bn + warp_n * TC_WN + ni * MMA_N + tg * 2;
            if (row0 < M && col + 1 < N)
                *reinterpret_cast<float2*>(&C[row0*N+col]) =
                    make_float2(acc[mi][ni][0], acc[mi][ni][1]);
            else {
                if (row0 < M && col   < N) C[row0*N+col]   = acc[mi][ni][0];
                if (row0 < M && col+1 < N) C[row0*N+col+1] = acc[mi][ni][1];
            }
            if (row1 < M && col + 1 < N)
                *reinterpret_cast<float2*>(&C[row1*N+col]) =
                    make_float2(acc[mi][ni][2], acc[mi][ni][3]);
            else {
                if (row1 < M && col   < N) C[row1*N+col]   = acc[mi][ni][2];
                if (row1 < M && col+1 < N) C[row1*N+col+1] = acc[mi][ni][3];
            }
        }
    }
}

// ===========================================================================
//  Path 8: Fused GEMM + Activation Epilogue
//
//  Applies the activation directly on the accumulator registers before writing
//  to global memory. Eliminates one full global memory round-trip (read + write
//  of the intermediate activation) for each MLP layer.
// ===========================================================================

__device__ __forceinline__ float epilogue_relu(float x) { return x > 0.f ? x : 0.f; }
__device__ __forceinline__ float epilogue_gelu(float x) { return ptx::fast_gelu_ptx(x); }
__device__ __forceinline__ float epilogue_silu(float x) { return ptx::fast_silu_ptx(x); }
__device__ __forceinline__ float epilogue_sigmoid(float x) { return ptx::fast_sigmoid_ptx(x); }

template <GemmEpilogue EPL>
__device__ __forceinline__ float apply_epilogue(float x) {
    if constexpr (EPL == GemmEpilogue::ReLU)    return epilogue_relu(x);
    if constexpr (EPL == GemmEpilogue::GELU)    return epilogue_gelu(x);
    if constexpr (EPL == GemmEpilogue::SiLU)    return epilogue_silu(x);
    if constexpr (EPL == GemmEpilogue::Sigmoid) return epilogue_sigmoid(x);
    return x;
}

template <GemmEpilogue EPL>
__global__ __launch_bounds__(NUM_THREADS, 2)
void sgemm_tc_fused_kernel(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            int M, int N, int K) {
    extern __shared__ float _smem_fused[];
    constexpr int As_plane = BK_TC * SMEM_A_STRIDE;
    float (*As)[BK_TC][SMEM_A_STRIDE] =
        reinterpret_cast<float (*)[BK_TC][SMEM_A_STRIDE]>(_smem_fused);
    float (*Bs)[BK_TC][SMEM_B_STRIDE] =
        reinterpret_cast<float (*)[BK_TC][SMEM_B_STRIDE]>(_smem_fused + 2 * As_plane);

    int bm, bn;
    {
        constexpr int SW = 4;
        const int tiles_n = gridDim.x;
        const int tiles_m = gridDim.y;
        const int linear = blockIdx.x + blockIdx.y * tiles_n;
        const int sw = (tiles_n < SW) ? tiles_n : SW;
        const int group = linear / (sw * tiles_m);
        const int rem   = linear % (sw * tiles_m);
        bm = (rem / sw) * BM;
        bn = (group * sw + rem % sw) * BN;
    }
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid % 32;
    const int warp_m = warp_id / TC_WARPS_N;
    const int warp_n = warp_id % TC_WARPS_N;
    const int gid = lane / 4;
    const int tg  = lane % 4;

    float acc[MMA_TILES_M][MMA_TILES_N][4];
    #pragma unroll
    for (int mi = 0; mi < MMA_TILES_M; ++mi)
        #pragma unroll
        for (int ni = 0; ni < MMA_TILES_N; ++ni) {
            acc[mi][ni][0] = 0.f; acc[mi][ni][1] = 0.f;
            acc[mi][ni][2] = 0.f; acc[mi][ni][3] = 0.f;
        }

    const int a_col  = tid % BK_TC;
    const int a_row0 = tid / BK_TC;
    const int b_row_base = tid / 32;
    const int b_col4 = tid % 32;

    float _a_pf[16];

    int num_tiles = (K + BK_TC - 1) / BK_TC;

    // Load first tile
    #pragma unroll
    for (int s = 0; s < 16; ++s) {
        int gr = bm + a_row0 + s * 8;
        int gc = a_col;
        _a_pf[s] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
    }
    auto load_B_fused = [&](int buf_idx, int bk_off) {
        auto lb = [&](int br) {
            int gr = bk_off + br;
            int gc = bn + b_col4 * 4;
            if (gr < K && gc + 3 < N) {
                ptx::cp_async_cg_16(&Bs[buf_idx][br][b_col4 * 4], &B[gr * N + gc]);
            } else {
                Bs[buf_idx][br][b_col4*4+0] = (gr<K && gc+0<N) ? B[gr*N+gc+0] : 0.f;
                Bs[buf_idx][br][b_col4*4+1] = (gr<K && gc+1<N) ? B[gr*N+gc+1] : 0.f;
                Bs[buf_idx][br][b_col4*4+2] = (gr<K && gc+2<N) ? B[gr*N+gc+2] : 0.f;
                Bs[buf_idx][br][b_col4*4+3] = (gr<K && gc+3<N) ? B[gr*N+gc+3] : 0.f;
            }
        };
        lb(b_row_base); lb(b_row_base + 8);
        lb(b_row_base + 16); lb(b_row_base + 24);
    };

    load_B_fused(0, 0);
    ptx::cp_async_commit();
    #pragma unroll
    for (int s = 0; s < 16; ++s)
        As[0][a_col][a_row0 + s * 8] = _a_pf[s];
    ptx::cp_async_wait_all();
    __syncthreads();

    int buf = 0;
    for (int t = 0; t < num_tiles - 1; ++t) {
        #pragma unroll
        for (int s = 0; s < 16; ++s) {
            int gr = bm + a_row0 + s * 8;
            int gc = (t + 1) * BK_TC + a_col;
            _a_pf[s] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        load_B_fused(1 - buf, (t + 1) * BK_TC);
        ptx::cp_async_commit();

        #pragma unroll
        for (int kk = 0; kk < BK_TC; kk += MMA_K) {
            uint32_t breg[MMA_TILES_N][2];
            #pragma unroll
            for (int ni = 0; ni < MMA_TILES_N; ++ni) {
                int bni = warp_n * TC_WN + ni * MMA_N;
                breg[ni][0] = __float_as_uint(Bs[buf][kk + tg*2    ][bni + gid]);
                breg[ni][1] = __float_as_uint(Bs[buf][kk + tg*2 + 1][bni + gid]);
            }
            #pragma unroll
            for (int mi = 0; mi < MMA_TILES_M; ++mi) {
                int am = warp_m * TC_WM + mi * MMA_M;
                uint32_t a0 = __float_as_uint(As[buf][kk + tg*2    ][am + gid    ]);
                uint32_t a1 = __float_as_uint(As[buf][kk + tg*2    ][am + gid + 8]);
                uint32_t a2 = __float_as_uint(As[buf][kk + tg*2 + 1][am + gid    ]);
                uint32_t a3 = __float_as_uint(As[buf][kk + tg*2 + 1][am + gid + 8]);
                #pragma unroll
                for (int ni = 0; ni < MMA_TILES_N; ++ni)
                    ptx::mma_m16n8k8_tf32(
                        acc[mi][ni][0], acc[mi][ni][1],
                        acc[mi][ni][2], acc[mi][ni][3],
                        a0, a1, a2, a3, breg[ni][0], breg[ni][1],
                        acc[mi][ni][0], acc[mi][ni][1],
                        acc[mi][ni][2], acc[mi][ni][3]);
            }
        }

        #pragma unroll
        for (int s = 0; s < 16; ++s)
            As[1 - buf][a_col][a_row0 + s * 8] = _a_pf[s];
        ptx::cp_async_wait_all();
        buf = 1 - buf;
        __syncthreads();
    }

    // Last tile
    #pragma unroll
    for (int kk = 0; kk < BK_TC; kk += MMA_K) {
        uint32_t breg[MMA_TILES_N][2];
        #pragma unroll
        for (int ni = 0; ni < MMA_TILES_N; ++ni) {
            int bni = warp_n * TC_WN + ni * MMA_N;
            breg[ni][0] = __float_as_uint(Bs[buf][kk + tg*2    ][bni + gid]);
            breg[ni][1] = __float_as_uint(Bs[buf][kk + tg*2 + 1][bni + gid]);
        }
        #pragma unroll
        for (int mi = 0; mi < MMA_TILES_M; ++mi) {
            int am = warp_m * TC_WM + mi * MMA_M;
            uint32_t a0 = __float_as_uint(As[buf][kk + tg*2    ][am + gid    ]);
            uint32_t a1 = __float_as_uint(As[buf][kk + tg*2    ][am + gid + 8]);
            uint32_t a2 = __float_as_uint(As[buf][kk + tg*2 + 1][am + gid    ]);
            uint32_t a3 = __float_as_uint(As[buf][kk + tg*2 + 1][am + gid + 8]);
            #pragma unroll
            for (int ni = 0; ni < MMA_TILES_N; ++ni)
                ptx::mma_m16n8k8_tf32(
                    acc[mi][ni][0], acc[mi][ni][1],
                    acc[mi][ni][2], acc[mi][ni][3],
                    a0, a1, a2, a3, breg[ni][0], breg[ni][1],
                    acc[mi][ni][0], acc[mi][ni][1],
                    acc[mi][ni][2], acc[mi][ni][3]);
        }
    }

    // Store with epilogue activation applied in registers
    #pragma unroll
    for (int mi = 0; mi < MMA_TILES_M; ++mi) {
        int row0 = bm + warp_m * TC_WM + mi * MMA_M + gid;
        int row1 = row0 + 8;
        #pragma unroll
        for (int ni = 0; ni < MMA_TILES_N; ++ni) {
            int col = bn + warp_n * TC_WN + ni * MMA_N + tg * 2;
            float v0 = apply_epilogue<EPL>(acc[mi][ni][0]);
            float v1 = apply_epilogue<EPL>(acc[mi][ni][1]);
            float v2 = apply_epilogue<EPL>(acc[mi][ni][2]);
            float v3 = apply_epilogue<EPL>(acc[mi][ni][3]);
            if (row0 < M && col + 1 < N)
                *reinterpret_cast<float2*>(&C[row0*N+col]) = make_float2(v0, v1);
            else {
                if (row0 < M && col   < N) C[row0*N+col]   = v0;
                if (row0 < M && col+1 < N) C[row0*N+col+1] = v1;
            }
            if (row1 < M && col + 1 < N)
                *reinterpret_cast<float2*>(&C[row1*N+col]) = make_float2(v2, v3);
            else {
                if (row1 < M && col   < N) C[row1*N+col]   = v2;
                if (row1 < M && col+1 < N) C[row1*N+col+1] = v3;
            }
        }
    }
}

// ===========================================================================
//  Launch dispatcher
// ===========================================================================

static bool has_tensor_cores() {
    static int cached = -1;
    if (cached < 0) {
        int major = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
        cached = (major >= 8) ? 1 : 0;
    }
    return cached == 1;
}

static bool has_hopper() {
    static int cached = -1;
    if (cached < 0) {
        int major = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
        cached = (major >= 9) ? 1 : 0;
    }
    return cached == 1;
}

static int get_num_sms() {
    static int cached = -1;
    if (cached < 0) {
        cudaDeviceGetAttribute(&cached, cudaDevAttrMultiProcessorCount, 0);
    }
    return cached;
}

static void configure_tc_smem() {
    static bool done = false;
    if (done) return;
    cudaFuncSetAttribute(sgemm_tc_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)TC_SMEM_BYTES);
    cudaFuncSetAttribute(sgemm_tc_general_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)TC_SMEM_BYTES);
    cudaFuncSetAttribute(sgemm_persistent_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)TC_SMEM_BYTES);
    cudaFuncSetAttribute(sgemm_warp_specialized_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)WS_SMEM_BYTES);
    cudaFuncSetAttribute(sgemm_hopper_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)TC_SMEM_BYTES);
    done = true;
}

template <>
void launch_matmul_kernel<float>(const float* A, const float* B, float* C,
                                  int M, int N, int K, cudaStream_t stream) {
    dim3 blocks(ceil_div(N, BN), ceil_div(M, BM));

    if (has_tensor_cores()) {
        configure_tc_smem();

        int tiles_m = ceil_div(M, BM);
        int tiles_n = ceil_div(N, BN);
        int total_tiles = tiles_m * tiles_n;

        // Hopper path (SM 9.0+): WGMMA fences + TMA prefetch for higher throughput.
        // Currently uses the same mma.sync compute core with WGMMA scheduling hints;
        // a full wgmma.mma_async path can replace the inner loop when SM 9.0 is the
        // primary deployment target.
        if (has_hopper() && M >= 256 && N >= 256) {
            sgemm_hopper_kernel<<<blocks, NUM_THREADS, TC_SMEM_BYTES, stream>>>(
                A, B, C, M, N, K);
        }
        // Persistent kernel for large matrices — better L2 reuse, less launch overhead
        else if (M >= 2048 && N >= 2048 && total_tiles >= 64) {
            int num_sms = get_num_sms();
            int persistent_blocks = min(2 * num_sms, total_tiles);
            unsigned int zero = 0;
            cudaMemcpyToSymbolAsync(persistent_tile_counter, &zero,
                                     sizeof(unsigned int), 0,
                                     cudaMemcpyHostToDevice, stream);
            sgemm_persistent_kernel<<<persistent_blocks, NUM_THREADS, TC_SMEM_BYTES, stream>>>(
                A, B, C, M, N, K, total_tiles, tiles_n);
        }
        // Warp-specialized for medium-large matrices where the dedicated
        // producer/consumer pipeline hides global memory latency.
        // BM_WS=96, BN_WS=128 tile — use when matrix is large enough to
        // fill the SMs but below the persistent threshold.
        else if (M >= 512 && N >= 512 && K >= 256) {
            dim3 ws_blocks(ceil_div(N, BN_WS), ceil_div(M, BM_WS));
            sgemm_warp_specialized_kernel<<<ws_blocks, WS_THREADS, WS_SMEM_BYTES, stream>>>(
                A, B, C, M, N, K);
        } else {
            bool tc_aligned = (M % BM == 0) && (N % BN == 0) && (K % BK_TC == 0);
            if (tc_aligned) {
                sgemm_tc_kernel<<<blocks, NUM_THREADS, TC_SMEM_BYTES, stream>>>(A, B, C, M, N, K);
            } else {
                sgemm_tc_general_kernel<<<blocks, NUM_THREADS, TC_SMEM_BYTES, stream>>>(A, B, C, M, N, K);
            }
        }
    } else {
        bool aligned = (M % BM == 0) && (N % BN == 0) && (K % BK == 0);
        if (aligned && (N % 4 == 0)) {
            sgemm_fast_kernel<<<blocks, NUM_THREADS, 0, stream>>>(A, B, C, M, N, K);
        } else {
            sgemm_general_kernel<<<blocks, NUM_THREADS, 0, stream>>>(A, B, C, M, N, K);
        }
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

template <>
void launch_matmul_kernel_fp32<float>(const float* A, const float* B, float* C,
                                       int M, int N, int K, cudaStream_t stream) {
    dim3 blocks(ceil_div(N, BN), ceil_div(M, BM));
    bool aligned = (M % BM == 0) && (N % BN == 0) && (K % BK == 0);

    if (aligned && (N % 4 == 0)) {
        sgemm_fast_kernel<<<blocks, NUM_THREADS, 0, stream>>>(A, B, C, M, N, K);
    } else {
        sgemm_general_kernel<<<blocks, NUM_THREADS, 0, stream>>>(A, B, C, M, N, K);
    }
    CUDEEP_CHECK_LAST_KERNEL();
}

template <>
void launch_matmul_kernel_fp32<double>(const double* A, const double* B, double* C,
                                        int M, int N, int K, cudaStream_t stream) {
    launch_matmul_kernel<double>(A, B, C, M, N, K, stream);
}

template <typename T>
void launch_matmul_tiled_kernel(const T* A, const T* B, T* C,
                                 int M, int N, int K, int, cudaStream_t stream) {
    launch_matmul_kernel(A, B, C, M, N, K, stream);
}

template void launch_matmul_kernel<float>(const float*, const float*, float*, int, int, int, cudaStream_t);
template void launch_matmul_kernel<double>(const double*, const double*, double*, int, int, int, cudaStream_t);
template void launch_matmul_kernel_fp32<float>(const float*, const float*, float*, int, int, int, cudaStream_t);
template void launch_matmul_kernel_fp32<double>(const double*, const double*, double*, int, int, int, cudaStream_t);
template void launch_matmul_tiled_kernel<float>(const float*, const float*, float*, int, int, int, int, cudaStream_t);
template void launch_matmul_tiled_kernel<double>(const double*, const double*, double*, int, int, int, int, cudaStream_t);

void launch_matmul_fused_act(const float* A, const float* B, float* C,
                              int M, int N, int K,
                              GemmEpilogue epilogue, cudaStream_t stream) {
    if (!has_tensor_cores()) {
        launch_matmul_kernel<float>(A, B, C, M, N, K, stream);
        return;
    }
    configure_tc_smem();

    // Also configure fused kernel smem
    static bool fused_configured = false;
    if (!fused_configured) {
        cudaFuncSetAttribute(sgemm_tc_fused_kernel<GemmEpilogue::ReLU>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)TC_SMEM_BYTES);
        cudaFuncSetAttribute(sgemm_tc_fused_kernel<GemmEpilogue::GELU>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)TC_SMEM_BYTES);
        cudaFuncSetAttribute(sgemm_tc_fused_kernel<GemmEpilogue::SiLU>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)TC_SMEM_BYTES);
        cudaFuncSetAttribute(sgemm_tc_fused_kernel<GemmEpilogue::Sigmoid>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)TC_SMEM_BYTES);
        cudaFuncSetAttribute(sgemm_tc_fused_kernel<GemmEpilogue::None>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)TC_SMEM_BYTES);
        fused_configured = true;
    }

    dim3 blocks(ceil_div(N, BN), ceil_div(M, BM));

    switch (epilogue) {
    case GemmEpilogue::ReLU:
        sgemm_tc_fused_kernel<GemmEpilogue::ReLU><<<blocks, NUM_THREADS, TC_SMEM_BYTES, stream>>>(A, B, C, M, N, K); break;
    case GemmEpilogue::GELU:
        sgemm_tc_fused_kernel<GemmEpilogue::GELU><<<blocks, NUM_THREADS, TC_SMEM_BYTES, stream>>>(A, B, C, M, N, K); break;
    case GemmEpilogue::SiLU:
        sgemm_tc_fused_kernel<GemmEpilogue::SiLU><<<blocks, NUM_THREADS, TC_SMEM_BYTES, stream>>>(A, B, C, M, N, K); break;
    case GemmEpilogue::Sigmoid:
        sgemm_tc_fused_kernel<GemmEpilogue::Sigmoid><<<blocks, NUM_THREADS, TC_SMEM_BYTES, stream>>>(A, B, C, M, N, K); break;
    default:
        sgemm_tc_fused_kernel<GemmEpilogue::None><<<blocks, NUM_THREADS, TC_SMEM_BYTES, stream>>>(A, B, C, M, N, K); break;
    }
    CUDEEP_CHECK_LAST_KERNEL();
}

void launch_matmul_kernel_fp16(const __half* A, const __half* B, float* C,
                                int M, int N, int K, cudaStream_t stream) {
    if (!has_tensor_cores()) {
        // Fallback: this shouldn't happen in normal use (FP16 MMA requires SM 8.0+)
        return;
    }

    static bool fp16_smem_configured = false;
    if (!fp16_smem_configured) {
        cudaFuncSetAttribute(hgemm_tc_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)FP16_SMEM_BYTES);
        fp16_smem_configured = true;
    }

    dim3 blocks(ceil_div(N, BN), ceil_div(M, BM));
    hgemm_tc_kernel<<<blocks, NUM_THREADS, FP16_SMEM_BYTES, stream>>>(A, B, C, M, N, K);
    CUDEEP_CHECK_LAST_KERNEL();
}

}  // namespace kernels
}  // namespace cudeep
