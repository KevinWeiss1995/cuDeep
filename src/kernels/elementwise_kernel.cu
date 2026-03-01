#include "cudeep/kernels/elementwise.cuh"
#include "cudeep/ptx_intrinsics.cuh"
#include "cudeep/error.cuh"

namespace cudeep {
namespace kernels {

// ---------------------------------------------------------------------------
// Vectorized elementwise kernels — float4 for float, double2 for double
//
// Each thread processes Vec4<T>::width elements at once (4 for float,
// 2 for double), maximizing memory bandwidth utilization.
// ---------------------------------------------------------------------------

// ---- Binary ops with vectorized path ----

template <typename T>
__global__ void add_kernel(const T* a, const T* b, T* out, int64_t n) {
    constexpr int W = Vec4<T>::width;
    int64_t base = (blockIdx.x * blockDim.x + threadIdx.x) * W;
    if (base + W <= n) {
        auto va = Vec4<T>::load(a + base);
        auto vb = Vec4<T>::load(b + base);
        if constexpr (W == 4) {
            float4 vc = {va.x + vb.x, va.y + vb.y, va.z + vb.z, va.w + vb.w};
            Vec4<T>::store(out + base, vc);
        } else {
            double2 vc = {va.x + vb.x, va.y + vb.y};
            Vec4<T>::store(out + base, vc);
        }
    } else {
        for (int64_t i = base; i < n && i < base + W; ++i)
            out[i] = a[i] + b[i];
    }
}

template <typename T>
__global__ void sub_kernel(const T* a, const T* b, T* out, int64_t n) {
    constexpr int W = Vec4<T>::width;
    int64_t base = (blockIdx.x * blockDim.x + threadIdx.x) * W;
    if (base + W <= n) {
        auto va = Vec4<T>::load(a + base);
        auto vb = Vec4<T>::load(b + base);
        if constexpr (W == 4) {
            float4 vc = {va.x - vb.x, va.y - vb.y, va.z - vb.z, va.w - vb.w};
            Vec4<T>::store(out + base, vc);
        } else {
            double2 vc = {va.x - vb.x, va.y - vb.y};
            Vec4<T>::store(out + base, vc);
        }
    } else {
        for (int64_t i = base; i < n && i < base + W; ++i)
            out[i] = a[i] - b[i];
    }
}

template <typename T>
__global__ void mul_kernel(const T* a, const T* b, T* out, int64_t n) {
    constexpr int W = Vec4<T>::width;
    int64_t base = (blockIdx.x * blockDim.x + threadIdx.x) * W;
    if (base + W <= n) {
        auto va = Vec4<T>::load(a + base);
        auto vb = Vec4<T>::load(b + base);
        if constexpr (W == 4) {
            float4 vc = {va.x * vb.x, va.y * vb.y, va.z * vb.z, va.w * vb.w};
            Vec4<T>::store(out + base, vc);
        } else {
            double2 vc = {va.x * vb.x, va.y * vb.y};
            Vec4<T>::store(out + base, vc);
        }
    } else {
        for (int64_t i = base; i < n && i < base + W; ++i)
            out[i] = a[i] * b[i];
    }
}

template <typename T>
__global__ void div_kernel(const T* a, const T* b, T* out, int64_t n) {
    constexpr int W = Vec4<T>::width;
    int64_t base = (blockIdx.x * blockDim.x + threadIdx.x) * W;
    if (base + W <= n) {
        auto va = Vec4<T>::load(a + base);
        auto vb = Vec4<T>::load(b + base);
        if constexpr (W == 4) {
            float4 vc = {va.x / vb.x, va.y / vb.y, va.z / vb.z, va.w / vb.w};
            Vec4<T>::store(out + base, vc);
        } else {
            double2 vc = {va.x / vb.x, va.y / vb.y};
            Vec4<T>::store(out + base, vc);
        }
    } else {
        for (int64_t i = base; i < n && i < base + W; ++i)
            out[i] = a[i] / b[i];
    }
}

// ---- Scalar multiply (vectorized) ----

template <typename T>
__global__ void scalar_mul_kernel(const T* a, float scalar, T* out, int64_t n) {
    constexpr int W = Vec4<T>::width;
    T sc = static_cast<T>(scalar);
    int64_t base = (blockIdx.x * blockDim.x + threadIdx.x) * W;
    if (base + W <= n) {
        auto va = Vec4<T>::load(a + base);
        if constexpr (W == 4) {
            float4 vc = {va.x * sc, va.y * sc, va.z * sc, va.w * sc};
            Vec4<T>::store(out + base, vc);
        } else {
            double2 vc = {va.x * sc, va.y * sc};
            Vec4<T>::store(out + base, vc);
        }
    } else {
        for (int64_t i = base; i < n && i < base + W; ++i)
            out[i] = a[i] * sc;
    }
}

// ---- Fill (vectorized) ----

template <typename T>
__global__ void fill_kernel(T* data, T value, int64_t n) {
    constexpr int W = Vec4<T>::width;
    int64_t base = (blockIdx.x * blockDim.x + threadIdx.x) * W;
    if (base + W <= n) {
        if constexpr (W == 4) {
            float4 v = {value, value, value, value};
            Vec4<T>::store(data + base, v);
        } else {
            double2 v = {value, value};
            Vec4<T>::store(data + base, v);
        }
    } else {
        for (int64_t i = base; i < n && i < base + W; ++i)
            data[i] = value;
    }
}

// ---- Launch wrappers (account for vectorized element count) ----

template <typename T>
static int vec_blocks(int64_t n) {
    constexpr int W = Vec4<T>::width;
    int64_t vec_n = (n + W - 1) / W;
    return ceil_div(static_cast<int>(vec_n), DEFAULT_BLOCK_SIZE);
}

template <typename T>
void launch_add_kernel(const T* a, const T* b, T* out, int64_t n, cudaStream_t stream) {
    add_kernel<<<vec_blocks<T>(n), DEFAULT_BLOCK_SIZE, 0, stream>>>(a, b, out, n);
    CUDEEP_CHECK_LAST_KERNEL();
}

template <typename T>
void launch_sub_kernel(const T* a, const T* b, T* out, int64_t n, cudaStream_t stream) {
    sub_kernel<<<vec_blocks<T>(n), DEFAULT_BLOCK_SIZE, 0, stream>>>(a, b, out, n);
    CUDEEP_CHECK_LAST_KERNEL();
}

template <typename T>
void launch_mul_kernel(const T* a, const T* b, T* out, int64_t n, cudaStream_t stream) {
    mul_kernel<<<vec_blocks<T>(n), DEFAULT_BLOCK_SIZE, 0, stream>>>(a, b, out, n);
    CUDEEP_CHECK_LAST_KERNEL();
}

template <typename T>
void launch_div_kernel(const T* a, const T* b, T* out, int64_t n, cudaStream_t stream) {
    div_kernel<<<vec_blocks<T>(n), DEFAULT_BLOCK_SIZE, 0, stream>>>(a, b, out, n);
    CUDEEP_CHECK_LAST_KERNEL();
}

template <typename T>
void launch_scalar_mul_kernel(const T* a, float scalar, T* out, int64_t n, cudaStream_t stream) {
    scalar_mul_kernel<<<vec_blocks<T>(n), DEFAULT_BLOCK_SIZE, 0, stream>>>(a, scalar, out, n);
    CUDEEP_CHECK_LAST_KERNEL();
}

template <typename T>
void launch_fill_kernel(T* data, T value, int64_t n, cudaStream_t stream) {
    fill_kernel<<<vec_blocks<T>(n), DEFAULT_BLOCK_SIZE, 0, stream>>>(data, value, n);
    CUDEEP_CHECK_LAST_KERNEL();
}

// ---- Broadcast: matrix [N,M] + row [M] (keep simple — memory-bound) ----

template <typename T>
__global__ void broadcast_add_row_kernel(const T* matrix, const T* row, T* output,
                                          int64_t rows, int64_t cols) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = rows * cols;
    if (idx >= total) return;
    int64_t col = idx % cols;
    output[idx] = matrix[idx] + row[col];
}

template <typename T>
void launch_broadcast_add_row_kernel(const T* matrix, const T* row, T* output,
                                      int64_t rows, int64_t cols, cudaStream_t stream) {
    int64_t total = rows * cols;
    int threads = DEFAULT_BLOCK_SIZE;
    int blocks = ceil_div(static_cast<int>(total), threads);
    broadcast_add_row_kernel<<<blocks, threads, 0, stream>>>(matrix, row, output, rows, cols);
    CUDEEP_CHECK_LAST_KERNEL();
}

// ---- Explicit instantiations ----
template void launch_add_kernel<float>(const float*, const float*, float*, int64_t, cudaStream_t);
template void launch_add_kernel<double>(const double*, const double*, double*, int64_t, cudaStream_t);
template void launch_sub_kernel<float>(const float*, const float*, float*, int64_t, cudaStream_t);
template void launch_sub_kernel<double>(const double*, const double*, double*, int64_t, cudaStream_t);
template void launch_mul_kernel<float>(const float*, const float*, float*, int64_t, cudaStream_t);
template void launch_mul_kernel<double>(const double*, const double*, double*, int64_t, cudaStream_t);
template void launch_div_kernel<float>(const float*, const float*, float*, int64_t, cudaStream_t);
template void launch_div_kernel<double>(const double*, const double*, double*, int64_t, cudaStream_t);
template void launch_scalar_mul_kernel<float>(const float*, float, float*, int64_t, cudaStream_t);
template void launch_scalar_mul_kernel<double>(const double*, float, double*, int64_t, cudaStream_t);
template void launch_fill_kernel<float>(float*, float, int64_t, cudaStream_t);
template void launch_fill_kernel<double>(double*, double, int64_t, cudaStream_t);
template void launch_broadcast_add_row_kernel<float>(const float*, const float*, float*, int64_t, int64_t, cudaStream_t);
template void launch_broadcast_add_row_kernel<double>(const double*, const double*, double*, int64_t, int64_t, cudaStream_t);

}  // namespace kernels
}  // namespace cudeep
