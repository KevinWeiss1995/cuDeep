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

}  // namespace cudeep
