#include "cudeep/kernels/conv.cuh"
#include "cudeep/error.cuh"

namespace cudeep {
namespace kernels {

// ---- Forward ----

template <typename T>
__global__ void conv2d_forward_kernel(
    const T* input, const T* weight, const T* bias, T* output,
    int batch, int in_channels, int out_channels,
    int in_h, int in_w,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int out_h, int out_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_channels * out_h * out_w;
    if (idx >= total) return;

    int ow = idx % out_w;
    int oh = (idx / out_w) % out_h;
    int oc = (idx / (out_w * out_h)) % out_channels;
    int b  = idx / (out_w * out_h * out_channels);

    T acc = bias ? bias[oc] : static_cast<T>(0);

    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int ih = oh * stride_h - pad_h + kh;
                int iw = ow * stride_w - pad_w + kw;

                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                    int in_idx = ((b * in_channels + ic) * in_h + ih) * in_w + iw;
                    int w_idx  = ((oc * in_channels + ic) * kernel_h + kh) * kernel_w + kw;
                    acc += input[in_idx] * weight[w_idx];
                }
            }
        }
    }

    output[idx] = acc;
}

template <typename T>
void launch_conv2d_forward_kernel(
    const T* input, const T* weight, const T* bias, T* output,
    int batch, int in_channels, int out_channels,
    int in_h, int in_w,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    cudaStream_t stream
) {
    int out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;
    int total = batch * out_channels * out_h * out_w;

    int threads = DEFAULT_BLOCK_SIZE;
    int blocks = ceil_div(total, threads);

    conv2d_forward_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, bias, output,
        batch, in_channels, out_channels,
        in_h, in_w, kernel_h, kernel_w,
        stride_h, stride_w, pad_h, pad_w,
        out_h, out_w
    );
    CUDEEP_CHECK_LAST_KERNEL();
}

// ---- Backward data ----
// Computes grad_input from grad_output and weight

template <typename T>
__global__ void conv2d_backward_data_kernel(
    const T* grad_output, const T* weight, T* grad_input,
    int batch, int in_channels, int out_channels,
    int in_h, int in_w,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int out_h, int out_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * in_channels * in_h * in_w;
    if (idx >= total) return;

    int iw = idx % in_w;
    int ih = (idx / in_w) % in_h;
    int ic = (idx / (in_w * in_h)) % in_channels;
    int b  = idx / (in_w * in_h * in_channels);

    T acc = static_cast<T>(0);

    for (int oc = 0; oc < out_channels; ++oc) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int oh_num = ih + pad_h - kh;
                int ow_num = iw + pad_w - kw;

                if (oh_num % stride_h == 0 && ow_num % stride_w == 0) {
                    int oh = oh_num / stride_h;
                    int ow = ow_num / stride_w;

                    if (oh >= 0 && oh < out_h && ow >= 0 && ow < out_w) {
                        int go_idx = ((b * out_channels + oc) * out_h + oh) * out_w + ow;
                        int w_idx  = ((oc * in_channels + ic) * kernel_h + kh) * kernel_w + kw;
                        acc += grad_output[go_idx] * weight[w_idx];
                    }
                }
            }
        }
    }

    grad_input[idx] = acc;
}

template <typename T>
void launch_conv2d_backward_data_kernel(
    const T* grad_output, const T* weight, T* grad_input,
    int batch, int in_channels, int out_channels,
    int in_h, int in_w,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    cudaStream_t stream
) {
    int out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;
    int total = batch * in_channels * in_h * in_w;

    int threads = DEFAULT_BLOCK_SIZE;
    int blocks = ceil_div(total, threads);

    conv2d_backward_data_kernel<<<blocks, threads, 0, stream>>>(
        grad_output, weight, grad_input,
        batch, in_channels, out_channels,
        in_h, in_w, kernel_h, kernel_w,
        stride_h, stride_w, pad_h, pad_w,
        out_h, out_w
    );
    CUDEEP_CHECK_LAST_KERNEL();
}

// ---- Backward weight ----
// Computes grad_weight from grad_output and input

template <typename T>
__global__ void conv2d_backward_weight_kernel(
    const T* grad_output, const T* input, T* grad_weight,
    int batch, int in_channels, int out_channels,
    int in_h, int in_w,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int out_h, int out_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = out_channels * in_channels * kernel_h * kernel_w;
    if (idx >= total) return;

    int kw = idx % kernel_w;
    int kh = (idx / kernel_w) % kernel_h;
    int ic = (idx / (kernel_w * kernel_h)) % in_channels;
    int oc = idx / (kernel_w * kernel_h * in_channels);

    T acc = static_cast<T>(0);

    for (int b = 0; b < batch; ++b) {
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                int ih = oh * stride_h - pad_h + kh;
                int iw = ow * stride_w - pad_w + kw;

                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                    int in_idx = ((b * in_channels + ic) * in_h + ih) * in_w + iw;
                    int go_idx = ((b * out_channels + oc) * out_h + oh) * out_w + ow;
                    acc += grad_output[go_idx] * input[in_idx];
                }
            }
        }
    }

    grad_weight[idx] = acc;
}

template <typename T>
void launch_conv2d_backward_weight_kernel(
    const T* grad_output, const T* input, T* grad_weight,
    int batch, int in_channels, int out_channels,
    int in_h, int in_w,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    cudaStream_t stream
) {
    int out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;
    int total = out_channels * in_channels * kernel_h * kernel_w;

    int threads = DEFAULT_BLOCK_SIZE;
    int blocks = ceil_div(total, threads);

    conv2d_backward_weight_kernel<<<blocks, threads, 0, stream>>>(
        grad_output, input, grad_weight,
        batch, in_channels, out_channels,
        in_h, in_w, kernel_h, kernel_w,
        stride_h, stride_w, pad_h, pad_w,
        out_h, out_w
    );
    CUDEEP_CHECK_LAST_KERNEL();
}

// ---- Explicit instantiations ----

template void launch_conv2d_forward_kernel<float>(const float*, const float*, const float*, float*, int, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
template void launch_conv2d_forward_kernel<double>(const double*, const double*, const double*, double*, int, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
template void launch_conv2d_backward_data_kernel<float>(const float*, const float*, float*, int, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
template void launch_conv2d_backward_data_kernel<double>(const double*, const double*, double*, int, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
template void launch_conv2d_backward_weight_kernel<float>(const float*, const float*, float*, int, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
template void launch_conv2d_backward_weight_kernel<double>(const double*, const double*, double*, int, int, int, int, int, int, int, int, int, int, int, cudaStream_t);

}  // namespace kernels
}  // namespace cudeep
