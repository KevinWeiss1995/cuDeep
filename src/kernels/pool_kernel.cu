#include "cudeep/kernels/pool.cuh"
#include "cudeep/error.cuh"

namespace cudeep {
namespace kernels {

// ---- MaxPool2d forward ----

template <typename T>
__global__ void maxpool2d_forward_kernel(
    const T* input, T* output,
    int batch, int channels,
    int in_h, int in_w,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int out_h, int out_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * out_h * out_w;
    if (idx >= total) return;

    int ow = idx % out_w;
    int oh = (idx / out_w) % out_h;
    int c  = (idx / (out_w * out_h)) % channels;
    int b  = idx / (out_w * out_h * channels);

    T max_val = static_cast<T>(-1e38);

    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            int ih = oh * stride_h - pad_h + kh;
            int iw = ow * stride_w - pad_w + kw;

            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                int in_idx = ((b * channels + c) * in_h + ih) * in_w + iw;
                max_val = max(max_val, input[in_idx]);
            }
        }
    }

    output[idx] = max_val;
}

template <typename T>
void launch_maxpool2d_forward_kernel(
    const T* input, T* output,
    int batch, int channels,
    int in_h, int in_w,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    cudaStream_t stream
) {
    int out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;
    int total = batch * channels * out_h * out_w;

    int threads = DEFAULT_BLOCK_SIZE;
    int blocks = ceil_div(total, threads);

    maxpool2d_forward_kernel<<<blocks, threads, 0, stream>>>(
        input, output,
        batch, channels, in_h, in_w,
        kernel_h, kernel_w, stride_h, stride_w,
        pad_h, pad_w, out_h, out_w);
    CUDEEP_CHECK_LAST_KERNEL();
}

// ---- AvgPool2d forward ----

template <typename T>
__global__ void avgpool2d_forward_kernel(
    const T* input, T* output,
    int batch, int channels,
    int in_h, int in_w,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int out_h, int out_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * out_h * out_w;
    if (idx >= total) return;

    int ow = idx % out_w;
    int oh = (idx / out_w) % out_h;
    int c  = (idx / (out_w * out_h)) % channels;
    int b  = idx / (out_w * out_h * channels);

    T sum = static_cast<T>(0);
    int count = 0;

    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            int ih = oh * stride_h - pad_h + kh;
            int iw = ow * stride_w - pad_w + kw;

            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                int in_idx = ((b * channels + c) * in_h + ih) * in_w + iw;
                sum += input[in_idx];
                ++count;
            }
        }
    }

    output[idx] = (count > 0) ? sum / static_cast<T>(count) : static_cast<T>(0);
}

template <typename T>
void launch_avgpool2d_forward_kernel(
    const T* input, T* output,
    int batch, int channels,
    int in_h, int in_w,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    cudaStream_t stream
) {
    int out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;
    int total = batch * channels * out_h * out_w;

    int threads = DEFAULT_BLOCK_SIZE;
    int blocks = ceil_div(total, threads);

    avgpool2d_forward_kernel<<<blocks, threads, 0, stream>>>(
        input, output,
        batch, channels, in_h, in_w,
        kernel_h, kernel_w, stride_h, stride_w,
        pad_h, pad_w, out_h, out_w);
    CUDEEP_CHECK_LAST_KERNEL();
}

// ---- Explicit instantiations ----

template void launch_maxpool2d_forward_kernel<float>(const float*, float*, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
template void launch_maxpool2d_forward_kernel<double>(const double*, double*, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
template void launch_avgpool2d_forward_kernel<float>(const float*, float*, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
template void launch_avgpool2d_forward_kernel<double>(const double*, double*, int, int, int, int, int, int, int, int, int, int, cudaStream_t);

}  // namespace kernels
}  // namespace cudeep
