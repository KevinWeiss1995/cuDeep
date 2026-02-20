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

// ---------------------------------------------------------------------------
// Warp-level reduction intrinsics (CUDA device code only)
// ---------------------------------------------------------------------------
#ifdef __CUDACC__

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
    int2 tmp = __double_as_longlong(val) >= 0
        ? make_int2(__double2loint(val), __double2hiint(val))
        : make_int2(__double2loint(val), __double2hiint(val));
    tmp.x = __shfl_down_sync(0xffffffff, tmp.x, offset);
    tmp.y = __shfl_down_sync(0xffffffff, tmp.y, offset);
    return __hiloint2double(tmp.y, tmp.x);
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

// Block-wide reduction using warp shuffles. Requires blockDim.x <= 1024.
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
// Vectorized load/store traits
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

#endif  // __CUDACC__

}  // namespace cudeep
