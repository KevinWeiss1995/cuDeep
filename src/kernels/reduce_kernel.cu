#include "cudeep/kernels/reduce.cuh"
#include "cudeep/ptx_intrinsics.cuh"
#include "cudeep/memory.cuh"
#include "cudeep/error.cuh"

#include <cfloat>

namespace cudeep {
namespace kernels {

// ---------------------------------------------------------------------------
// Optimized reductions: grid-stride loop + warp shuffle
//
// Each thread accumulates multiple elements via grid-stride loop,
// then does a warp-shuffle reduction, then a single shared-memory step
// across warps. Much fewer __syncthreads() and better throughput.
// ---------------------------------------------------------------------------

// ---- Sum ----

template <typename T>
__global__ void sum_kernel(const T* input, T* output, int64_t n) {
    T acc = T(0);
    constexpr int VW = Vec4<T>::width;
    int64_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    int64_t vec_n  = n / VW;

    for (int64_t i = tid; i < vec_n; i += stride) {
        auto v = Vec4<T>::load(&input[i * VW]);
        if constexpr (sizeof(T) == 4) { acc += v.x + v.y + v.z + v.w; }
        else                           { acc += v.x + v.y; }
    }
    for (int64_t i = vec_n * VW + tid; i < n; i += stride)
        acc += input[i];

    acc = block_reduce_sum(acc);

    if (threadIdx.x == 0)
        atomicAdd(output, acc);
}

// ===========================================================================
// Tensor Core Reduction (SM 8.0+, float only, speculative)
//
// For large arrays (10M+ elements): reshape the 1D vector into a 2D matrix
// [rows, TC_RED_K] and use TF32 MMA to compute row sums by multiplying
// against a shared-memory buffer of all ones.  Each mma.sync.m16n8k8
// processes 128 multiply-adds in one instruction — much higher throughput
// than scalar add chains.
//
// Precision: TF32 truncates mantissa from 23 to 10 bits, so each element
// is rounded before accumulation.  Acceptable for gradient sums but NOT
// for double-precision or loss computation.
// ===========================================================================

constexpr int TC_RED_K = 128;     // columns per row = elements summed via MMA
constexpr int TC_RED_M = 128;     // rows per block
constexpr int TC_RED_CHUNK = TC_RED_M * TC_RED_K;  // 16K elements per block
constexpr int TC_RED_THREADS = 256;
constexpr int TC_RED_MMA_M = 16;
constexpr int TC_RED_MMA_K = 8;
constexpr int TC_RED_MMA_N = 8;
constexpr int TC_RED_MTILES = TC_RED_M / TC_RED_MMA_M;   // 8
constexpr int TC_RED_KSTEPS = TC_RED_K / TC_RED_MMA_K;   // 16

constexpr int TC_RED_AS_STRIDE = TC_RED_M + 4;  // pad for bank conflicts

__global__ __launch_bounds__(TC_RED_THREADS)
void sum_tc_kernel(const float* __restrict__ input, float* output, int64_t n) {
    // Shared memory: As[k][m] (transposed input tile), Bs[k][n] (all ones)
    __shared__ float As[TC_RED_MMA_K][TC_RED_AS_STRIDE];
    __shared__ float Bs[TC_RED_MMA_K][TC_RED_MMA_N + 4];

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid % 32;
    const int gid = lane / 4;
    const int tg  = lane % 4;

    // Fill Bs with ones (each thread contributes)
    if (tid < TC_RED_MMA_K * TC_RED_MMA_N) {
        int bk = tid / TC_RED_MMA_N;
        int bn = tid % TC_RED_MMA_N;
        Bs[bk][bn] = 1.0f;
    }
    __syncthreads();

    // Pre-load B registers (constant across all tiles)
    uint32_t breg[2];
    breg[0] = __float_as_uint(Bs[tg * 2    ][gid]);
    breg[1] = __float_as_uint(Bs[tg * 2 + 1][gid]);

    // Each block processes one or more TC_RED_CHUNK-sized chunks
    float block_sum = 0.0f;
    int64_t block_start = (int64_t)blockIdx.x * TC_RED_CHUNK;
    const float* chunk = input + block_start;
    int64_t chunk_len = min((int64_t)TC_RED_CHUNK, n - block_start);
    if (chunk_len <= 0) return;

    int rows_this_block = (int)((chunk_len + TC_RED_K - 1) / TC_RED_K);

    // Each warp processes a subset of m-tiles
    // 8 warps, 8 m-tiles → 1 m-tile per warp (16 rows)
    int mi = warp_id;
    if (mi >= TC_RED_MTILES) mi = -1;  // excess warps idle during MMA

    float acc[4] = {0.f, 0.f, 0.f, 0.f};

    if (mi >= 0) {
        int m_base = mi * TC_RED_MMA_M;

        for (int kstep = 0; kstep < TC_RED_KSTEPS; ++kstep) {
            // Cooperative load: all threads in block load one k-slice of As
            // As[k][m] = chunk[m * TC_RED_K + (kstep * MMA_K + k)]
            __syncthreads();
            for (int idx = tid; idx < TC_RED_MMA_K * TC_RED_M; idx += TC_RED_THREADS) {
                int k = idx / TC_RED_M;
                int m = idx % TC_RED_M;
                int64_t flat = (int64_t)m * TC_RED_K + kstep * TC_RED_MMA_K + k;
                As[k][m] = (m < rows_this_block && flat < chunk_len)
                           ? chunk[flat] : 0.0f;
            }
            __syncthreads();

            uint32_t a0 = __float_as_uint(As[tg * 2    ][m_base + gid    ]);
            uint32_t a1 = __float_as_uint(As[tg * 2    ][m_base + gid + 8]);
            uint32_t a2 = __float_as_uint(As[tg * 2 + 1][m_base + gid    ]);
            uint32_t a3 = __float_as_uint(As[tg * 2 + 1][m_base + gid + 8]);

            ptx::mma_m16n8k8_tf32(acc[0], acc[1], acc[2], acc[3],
                                   a0, a1, a2, a3,
                                   breg[0], breg[1],
                                   acc[0], acc[1], acc[2], acc[3]);
        }

        // acc now holds row sums: acc[0] = row(m_base+gid), acc[2] = row(m_base+gid+8)
        // acc[1] and acc[3] are duplicates (all B columns were ones)
        // Sum the 16 row sums within this warp
        float warp_sum = acc[0] + acc[2];  // two rows per lane
        warp_sum = warp_reduce_sum(warp_sum);
        if (lane == 0)
            block_sum = warp_sum;
    }

    // Combine across warps
    __shared__ float warp_sums[32];
    if (mi >= 0 && lane == 0)
        warp_sums[warp_id] = block_sum;
    else if (mi < 0 && lane == 0)
        warp_sums[warp_id] = 0.0f;
    __syncthreads();

    if (warp_id == 0) {
        float val = (tid < TC_RED_MTILES) ? warp_sums[tid] : 0.0f;
        val = warp_reduce_sum(val);
        if (tid == 0)
            atomicAdd(output, val);
    }
}

static bool has_tensor_cores_reduce() {
    static int cached = -1;
    if (cached < 0) {
        int major = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
        cached = (major >= 8) ? 1 : 0;
    }
    return cached == 1;
}

template <typename T>
void launch_sum_kernel(const T* input, T* output, int64_t n, cudaStream_t stream) {
    CUDEEP_CHECK_CUDA(cudaMemsetAsync(output, 0, sizeof(T), stream));

    if constexpr (sizeof(T) == 4) {
        if (has_tensor_cores_reduce() && n >= 10'000'000) {
            int blocks = static_cast<int>((n + TC_RED_CHUNK - 1) / TC_RED_CHUNK);
            blocks = min(blocks, 4096);
            sum_tc_kernel<<<blocks, TC_RED_THREADS, 0, stream>>>(
                reinterpret_cast<const float*>(input),
                reinterpret_cast<float*>(output), n);
            CUDEEP_CHECK_LAST_KERNEL();
            return;
        }
    }

    int threads = DEFAULT_BLOCK_SIZE;
    int blocks = min(ceil_div(static_cast<int>(n), threads), 256);
    sum_kernel<<<blocks, threads, 0, stream>>>(input, output, n);
    CUDEEP_CHECK_LAST_KERNEL();
}

// ---- Mean ----

template <typename T>
__global__ void divide_scalar_kernel(T* val, T divisor) {
    if (threadIdx.x == 0) {
        if constexpr (sizeof(T) == 4)
            *val *= ptx::sfu_rcp(static_cast<float>(divisor));
        else
            *val /= divisor;
    }
}

template <typename T>
void launch_mean_kernel(const T* input, T* output, int64_t n, cudaStream_t stream) {
    if (n == 0) {
        CUDEEP_CHECK_CUDA(cudaMemsetAsync(output, 0, sizeof(T), stream));
        return;
    }
    launch_sum_kernel(input, output, n, stream);
    divide_scalar_kernel<<<1, 1, 0, stream>>>(output, static_cast<T>(n));
    CUDEEP_CHECK_LAST_KERNEL();
}

// ---- Max ----

__device__ void atomicMaxFloat(float* addr, float val) {
    int* addr_int = reinterpret_cast<int*>(addr);
    int old = *addr_int, assumed;
    do {
        assumed = old;
        if (__int_as_float(assumed) >= val) break;
        old = atomicCAS(addr_int, assumed, __float_as_int(val));
    } while (assumed != old);
}

__device__ void atomicMaxDouble(double* addr, double val) {
    unsigned long long int* addr_ull = reinterpret_cast<unsigned long long int*>(addr);
    unsigned long long int old = *addr_ull, assumed;
    do {
        assumed = old;
        if (__longlong_as_double(assumed) >= val) break;
        old = atomicCAS(addr_ull, assumed, __double_as_longlong(val));
    } while (assumed != old);
}

template <typename T>
__global__ void max_kernel(const T* input, T* output, int64_t n) {
    T val = T(-1e38);
    constexpr int VW = Vec4<T>::width;
    int64_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    int64_t vec_n  = n / VW;

    for (int64_t i = tid; i < vec_n; i += stride) {
        auto v = Vec4<T>::load(&input[i * VW]);
        if constexpr (sizeof(T) == 4) {
            val = max(val, max(max(v.x, v.y), max(v.z, v.w)));
        } else {
            val = max(val, max(v.x, v.y));
        }
    }
    for (int64_t i = vec_n * VW + tid; i < n; i += stride)
        val = max(val, input[i]);

    val = block_reduce_max(val);

    if (threadIdx.x == 0)
        atomicMaxFloat(reinterpret_cast<float*>(output), static_cast<float>(val));
}

template <>
__global__ void max_kernel<double>(const double* input, double* output, int64_t n) {
    double val = -1e308;
    int64_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    int64_t vec_n  = n / 2;

    for (int64_t i = tid; i < vec_n; i += stride) {
        double2 v = Vec4<double>::load(&input[i * 2]);
        val = fmax(val, fmax(v.x, v.y));
    }
    for (int64_t i = vec_n * 2 + tid; i < n; i += stride)
        val = fmax(val, input[i]);

    val = block_reduce_max(val);

    if (threadIdx.x == 0)
        atomicMaxDouble(output, val);
}

template <typename T>
void launch_max_kernel(const T* input, T* output, int64_t n, cudaStream_t stream) {
    int threads = DEFAULT_BLOCK_SIZE;
    int blocks = min(ceil_div(static_cast<int>(n), threads), 256);

    // Initialize to -inf
    T neg_inf;
    if constexpr (sizeof(T) == 4)
        neg_inf = -1e38f;
    else
        neg_inf = -1e308;
    CUDEEP_CHECK_CUDA(cudaMemcpyAsync(output, &neg_inf, sizeof(T),
                                       cudaMemcpyHostToDevice, stream));
    max_kernel<<<blocks, threads, 0, stream>>>(input, output, n);
    CUDEEP_CHECK_LAST_KERNEL();
}

// ---- Min ----

__device__ void atomicMinFloat(float* addr, float val) {
    int* addr_int = reinterpret_cast<int*>(addr);
    int old = *addr_int, assumed;
    do {
        assumed = old;
        if (__int_as_float(assumed) <= val) break;
        old = atomicCAS(addr_int, assumed, __float_as_int(val));
    } while (assumed != old);
}

__device__ void atomicMinDouble(double* addr, double val) {
    unsigned long long int* addr_ull = reinterpret_cast<unsigned long long int*>(addr);
    unsigned long long int old = *addr_ull, assumed;
    do {
        assumed = old;
        if (__longlong_as_double(assumed) <= val) break;
        old = atomicCAS(addr_ull, assumed, __double_as_longlong(val));
    } while (assumed != old);
}

template <typename T>
__global__ void min_kernel(const T* input, T* output, int64_t n) {
    T val = T(1e38);
    constexpr int VW = Vec4<T>::width;
    int64_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    int64_t vec_n  = n / VW;

    for (int64_t i = tid; i < vec_n; i += stride) {
        auto v = Vec4<T>::load(&input[i * VW]);
        if constexpr (sizeof(T) == 4) {
            val = min(val, min(min(v.x, v.y), min(v.z, v.w)));
        } else {
            val = min(val, min(v.x, v.y));
        }
    }
    for (int64_t i = vec_n * VW + tid; i < n; i += stride)
        val = min(val, input[i]);

    val = block_reduce_min(val);

    if (threadIdx.x == 0)
        atomicMinFloat(reinterpret_cast<float*>(output), static_cast<float>(val));
}

template <>
__global__ void min_kernel<double>(const double* input, double* output, int64_t n) {
    double val = 1e308;
    int64_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    int64_t vec_n  = n / 2;

    for (int64_t i = tid; i < vec_n; i += stride) {
        double2 v = Vec4<double>::load(&input[i * 2]);
        val = fmin(val, fmin(v.x, v.y));
    }
    for (int64_t i = vec_n * 2 + tid; i < n; i += stride)
        val = fmin(val, input[i]);

    val = block_reduce_min(val);

    if (threadIdx.x == 0)
        atomicMinDouble(output, val);
}

template <typename T>
void launch_min_kernel(const T* input, T* output, int64_t n, cudaStream_t stream) {
    int threads = DEFAULT_BLOCK_SIZE;
    int blocks = min(ceil_div(static_cast<int>(n), threads), 256);

    T pos_inf;
    if constexpr (sizeof(T) == 4)
        pos_inf = 1e38f;
    else
        pos_inf = 1e308;
    CUDEEP_CHECK_CUDA(cudaMemcpyAsync(output, &pos_inf, sizeof(T),
                                       cudaMemcpyHostToDevice, stream));
    min_kernel<<<blocks, threads, 0, stream>>>(input, output, n);
    CUDEEP_CHECK_LAST_KERNEL();
}

// ---- Sum along axis ----

template <typename T>
__global__ void sum_along_axis_kernel(
    const T* input, T* output,
    int64_t outer_size, int64_t axis_size, int64_t inner_size
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_out = outer_size * inner_size;
    if (idx >= total_out) return;

    int64_t outer = idx / inner_size;
    int64_t inner = idx % inner_size;

    T acc = T(0);
    for (int64_t a = 0; a < axis_size; ++a) {
        acc += input[(outer * axis_size + a) * inner_size + inner];
    }
    output[idx] = acc;
}

template <typename T>
void launch_sum_along_axis_kernel(
    const T* input, T* output,
    const int64_t* shape, int ndim, int axis,
    cudaStream_t stream
) {
    int64_t outer_size = 1;
    for (int i = 0; i < axis; ++i) outer_size *= shape[i];
    int64_t axis_size = shape[axis];
    int64_t inner_size = 1;
    for (int i = axis + 1; i < ndim; ++i) inner_size *= shape[i];

    int64_t total_out = outer_size * inner_size;
    int threads = DEFAULT_BLOCK_SIZE;
    int blocks = ceil_div(static_cast<int>(total_out), threads);

    sum_along_axis_kernel<<<blocks, threads, 0, stream>>>(
        input, output, outer_size, axis_size, inner_size);
    CUDEEP_CHECK_LAST_KERNEL();
}

// ---- Explicit instantiations ----

template void launch_sum_kernel<float>(const float*, float*, int64_t, cudaStream_t);
template void launch_sum_kernel<double>(const double*, double*, int64_t, cudaStream_t);
template void launch_mean_kernel<float>(const float*, float*, int64_t, cudaStream_t);
template void launch_mean_kernel<double>(const double*, double*, int64_t, cudaStream_t);
template void launch_max_kernel<float>(const float*, float*, int64_t, cudaStream_t);
template void launch_max_kernel<double>(const double*, double*, int64_t, cudaStream_t);
template void launch_min_kernel<float>(const float*, float*, int64_t, cudaStream_t);
template void launch_min_kernel<double>(const double*, double*, int64_t, cudaStream_t);
template void launch_sum_along_axis_kernel<float>(const float*, float*, const int64_t*, int, int, cudaStream_t);
template void launch_sum_along_axis_kernel<double>(const double*, double*, const int64_t*, int, int, cudaStream_t);

}  // namespace kernels
}  // namespace cudeep
