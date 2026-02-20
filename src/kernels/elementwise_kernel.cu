#include "cudeep/kernels/elementwise.cuh"
#include "cudeep/error.cuh"

namespace cudeep {
namespace kernels {

template <typename T>
__global__ void add_kernel(const T* a, const T* b, T* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

template <typename T>
__global__ void sub_kernel(const T* a, const T* b, T* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] - b[idx];
    }
}

template <typename T>
__global__ void mul_kernel(const T* a, const T* b, T* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * b[idx];
    }
}

template <typename T>
__global__ void div_kernel(const T* a, const T* b, T* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] / b[idx];
    }
}

template <typename T>
__global__ void scalar_mul_kernel(const T* a, float scalar, T* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * static_cast<T>(scalar);
    }
}

template <typename T>
__global__ void fill_kernel(T* data, T value, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = value;
    }
}

// --- Launch wrappers ---

template <typename T>
void launch_add_kernel(const T* a, const T* b, T* out, int64_t n, cudaStream_t stream) {
    int threads = DEFAULT_BLOCK_SIZE;
    int blocks = ceil_div(static_cast<int>(n), threads);
    add_kernel<<<blocks, threads, 0, stream>>>(a, b, out, n);
    CUDEEP_CHECK_LAST_KERNEL();
}

template <typename T>
void launch_sub_kernel(const T* a, const T* b, T* out, int64_t n, cudaStream_t stream) {
    int threads = DEFAULT_BLOCK_SIZE;
    int blocks = ceil_div(static_cast<int>(n), threads);
    sub_kernel<<<blocks, threads, 0, stream>>>(a, b, out, n);
    CUDEEP_CHECK_LAST_KERNEL();
}

template <typename T>
void launch_mul_kernel(const T* a, const T* b, T* out, int64_t n, cudaStream_t stream) {
    int threads = DEFAULT_BLOCK_SIZE;
    int blocks = ceil_div(static_cast<int>(n), threads);
    mul_kernel<<<blocks, threads, 0, stream>>>(a, b, out, n);
    CUDEEP_CHECK_LAST_KERNEL();
}

template <typename T>
void launch_div_kernel(const T* a, const T* b, T* out, int64_t n, cudaStream_t stream) {
    int threads = DEFAULT_BLOCK_SIZE;
    int blocks = ceil_div(static_cast<int>(n), threads);
    div_kernel<<<blocks, threads, 0, stream>>>(a, b, out, n);
    CUDEEP_CHECK_LAST_KERNEL();
}

template <typename T>
void launch_scalar_mul_kernel(const T* a, float scalar, T* out, int64_t n, cudaStream_t stream) {
    int threads = DEFAULT_BLOCK_SIZE;
    int blocks = ceil_div(static_cast<int>(n), threads);
    scalar_mul_kernel<<<blocks, threads, 0, stream>>>(a, scalar, out, n);
    CUDEEP_CHECK_LAST_KERNEL();
}

template <typename T>
void launch_fill_kernel(T* data, T value, int64_t n, cudaStream_t stream) {
    int threads = DEFAULT_BLOCK_SIZE;
    int blocks = ceil_div(static_cast<int>(n), threads);
    fill_kernel<<<blocks, threads, 0, stream>>>(data, value, n);
    CUDEEP_CHECK_LAST_KERNEL();
}

// ---- Broadcast: matrix [N,M] + row [M] ----

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

// Explicit instantiations
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
