#include "cudeep/kernels/reduce.cuh"
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
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x) {
        acc += input[i];
    }

    acc = block_reduce_sum(acc);

    if (threadIdx.x == 0)
        atomicAdd(output, acc);
}

template <typename T>
void launch_sum_kernel(const T* input, T* output, int64_t n, cudaStream_t stream) {
    CUDEEP_CHECK_CUDA(cudaMemsetAsync(output, 0, sizeof(T), stream));
    int threads = DEFAULT_BLOCK_SIZE;
    int blocks = min(ceil_div(static_cast<int>(n), threads), 256);
    sum_kernel<<<blocks, threads, 0, stream>>>(input, output, n);
    CUDEEP_CHECK_LAST_KERNEL();
}

// ---- Mean ----

template <typename T>
__global__ void divide_scalar_kernel(T* val, T divisor) {
    if (threadIdx.x == 0)
        *val /= divisor;
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
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x) {
        val = max(val, input[i]);
    }

    val = block_reduce_max(val);

    if (threadIdx.x == 0)
        atomicMaxFloat(reinterpret_cast<float*>(output), static_cast<float>(val));
}

template <>
__global__ void max_kernel<double>(const double* input, double* output, int64_t n) {
    double val = -1e308;
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x) {
        val = fmax(val, input[i]);
    }

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
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x) {
        val = min(val, input[i]);
    }

    val = block_reduce_min(val);

    if (threadIdx.x == 0)
        atomicMinFloat(reinterpret_cast<float*>(output), static_cast<float>(val));
}

template <>
__global__ void min_kernel<double>(const double* input, double* output, int64_t n) {
    double val = 1e308;
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x) {
        val = fmin(val, input[i]);
    }

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
