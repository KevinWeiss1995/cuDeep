#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

namespace cudeep {

enum class DType {
    Float16,
    Float32,
    Float64
};

enum class Layout {
    NCHW,
    NHWC
};

constexpr int WARP_SIZE = 32;
constexpr int MAX_THREADS_PER_BLOCK = 1024;
constexpr int DEFAULT_BLOCK_SIZE = 256;

template <DType dtype>
struct DTypeTraits;

template <>
struct DTypeTraits<DType::Float16> {
    using type = __half;
    static constexpr size_t size = 2;
};

template <>
struct DTypeTraits<DType::Float32> {
    using type = float;
    static constexpr size_t size = 4;
};

template <>
struct DTypeTraits<DType::Float64> {
    using type = double;
    static constexpr size_t size = 8;
};

inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

inline int64_t ceil_div64(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

// ---------------------------------------------------------------------------
// CUDA device code only
// ---------------------------------------------------------------------------
#ifdef __CUDACC__

// ---------------------------------------------------------------------------
// Warp-level reduction intrinsics
// ---------------------------------------------------------------------------

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__device__ __forceinline__ float warp_reduce_min(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val = fminf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__device__ __forceinline__ double shfl_down_double(double val, int offset) {
    int lo = __double2loint(val);
    int hi = __double2hiint(val);
    lo = __shfl_down_sync(0xffffffff, lo, offset);
    hi = __shfl_down_sync(0xffffffff, hi, offset);
    return __hiloint2double(hi, lo);
}

__device__ __forceinline__ double warp_reduce_sum(double val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += shfl_down_double(val, offset);
    return val;
}

__device__ __forceinline__ double warp_reduce_max(double val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val = fmax(val, shfl_down_double(val, offset));
    return val;
}

__device__ __forceinline__ double warp_reduce_min(double val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val = fmin(val, shfl_down_double(val, offset));
    return val;
}

// ---------------------------------------------------------------------------
// Block-wide reductions using warp shuffles. blockDim.x <= 1024.
// ---------------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ T block_reduce_sum(T val) {
    __shared__ T warp_sums[32];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    val = warp_reduce_sum(val);
    if (lane == 0) warp_sums[wid] = val;
    __syncthreads();

    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    val = (threadIdx.x < num_warps) ? warp_sums[threadIdx.x] : T(0);
    if (wid == 0) val = warp_reduce_sum(val);
    return val;
}

template <typename T>
__device__ __forceinline__ T block_reduce_max(T val) {
    __shared__ T warp_maxs[32];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    val = warp_reduce_max(val);
    if (lane == 0) warp_maxs[wid] = val;
    __syncthreads();

    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    val = (threadIdx.x < num_warps) ? warp_maxs[threadIdx.x] : T(-1e38);
    if (wid == 0) val = warp_reduce_max(val);
    return val;
}

template <typename T>
__device__ __forceinline__ T block_reduce_min(T val) {
    __shared__ T warp_mins[32];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    val = warp_reduce_min(val);
    if (lane == 0) warp_mins[wid] = val;
    __syncthreads();

    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    val = (threadIdx.x < num_warps) ? warp_mins[threadIdx.x] : T(1e38);
    if (wid == 0) val = warp_reduce_min(val);
    return val;
}

// ---------------------------------------------------------------------------
// Vectorized load/store traits — 128-bit aligned
// ---------------------------------------------------------------------------

template <typename T> struct Vec4;
template <> struct Vec4<float> {
    using type = float4;
    static constexpr int width = 4;
    static __device__ __forceinline__ float4 load(const float* p) {
        return *reinterpret_cast<const float4*>(p);
    }
    static __device__ __forceinline__ void store(float* p, float4 v) {
        *reinterpret_cast<float4*>(p) = v;
    }
};
template <> struct Vec4<double> {
    using type = double2;
    static constexpr int width = 2;
    static __device__ __forceinline__ double2 load(const double* p) {
        return *reinterpret_cast<const double2*>(p);
    }
    static __device__ __forceinline__ void store(double* p, double2 v) {
        *reinterpret_cast<double2*>(p) = v;
    }
};

// ---------------------------------------------------------------------------
// Async copy helpers (SM 80+ / Ampere+)
// Uses cp.async to copy directly from global → shared memory, bypassing
// the register file. Dramatically reduces register pressure and enables
// true overlap of global loads with shared memory computation.
// ---------------------------------------------------------------------------

#if __CUDA_ARCH__ >= 800

__device__ __forceinline__ void cp_async_f4(void* smem_dst, const void* global_src) {
    uint32_t smem_addr = static_cast<uint32_t>(
        __cvta_generic_to_shared(smem_dst));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(global_src) : "memory"
    );
}

__device__ __forceinline__ void cp_async_f4_zfill(
    void* smem_dst, const void* global_src, bool pred
) {
    uint32_t smem_addr = static_cast<uint32_t>(
        __cvta_generic_to_shared(smem_dst));
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %2, 0;\n"
        "  @!p st.shared.v4.b32 [%0], {0, 0, 0, 0};\n"
        "  @p cp.async.cg.shared.global [%0], [%1], 16;\n"
        "}\n"
        :: "r"(smem_addr), "l"(global_src), "r"((int)pred) : "memory"
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

#else

__device__ __forceinline__ void cp_async_commit() {}
__device__ __forceinline__ void cp_async_wait_all() {}
template <int N>
__device__ __forceinline__ void cp_async_wait_group() {}

#endif  // __CUDA_ARCH__ >= 800

// ---------------------------------------------------------------------------
// Fast math wrappers — use hardware SFU where available
// ---------------------------------------------------------------------------

__device__ __forceinline__ float fast_exp(float x) { return __expf(x); }
__device__ __forceinline__ float fast_tanh(float x) {
    // tanhf is already fast under --use_fast_math and handles overflow
    return tanhf(x);
}
__device__ __forceinline__ float fast_sigmoid(float x) {
    return 1.0f / (1.0f + __expf(-x));
}
__device__ __forceinline__ float fast_rsqrt(float x) { return rsqrtf(x); }

#endif  // __CUDACC__

}  // namespace cudeep
