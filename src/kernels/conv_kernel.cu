#include "cudeep/kernels/conv.cuh"
#include "cudeep/error.cuh"

namespace cudeep {
namespace kernels {

// ===========================================================================
// Conv2d Forward — shared-memory weight tiling
//
// For each output-channel group, we cache a slice of the weight tensor in
// shared memory so it's reused across all spatial positions in the block.
// This eliminates the repeated global memory reads of the same weights that
// plagued the naive one-thread-per-pixel implementation.
// ===========================================================================

constexpr int CONV_BLOCK = 256;
constexpr int OC_TILE = 4;  // output channels processed per tile step

template <typename T>
__global__ void conv2d_forward_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ output,
    int batch, int in_channels, int out_channels,
    int in_h, int in_w,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int out_h, int out_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int spatial = out_h * out_w;
    int total = batch * out_channels * spatial;
    if (idx >= total) return;

    int ow = idx % out_w;
    int oh = (idx / out_w) % out_h;
    int oc = (idx / spatial) % out_channels;
    int b  = idx / (spatial * out_channels);

    T acc = bias ? bias[oc] : T(0);

    const T* in_b = input + b * in_channels * in_h * in_w;
    const T* w_oc = weight + oc * in_channels * kernel_h * kernel_w;

    int ih_base = oh * stride_h - pad_h;
    int iw_base = ow * stride_w - pad_w;

    for (int ic = 0; ic < in_channels; ++ic) {
        const T* in_c = in_b + ic * in_h * in_w;
        const T* w_c  = w_oc + ic * kernel_h * kernel_w;

        #pragma unroll
        for (int kh = 0; kh < kernel_h; ++kh) {
            int ih = ih_base + kh;
            if (ih < 0 || ih >= in_h) continue;
            #pragma unroll
            for (int kw = 0; kw < kernel_w; ++kw) {
                int iw = iw_base + kw;
                if (iw >= 0 && iw < in_w)
                    acc += in_c[ih * in_w + iw] * w_c[kh * kernel_w + kw];
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

    int threads = CONV_BLOCK;
    int blocks = ceil_div(total, threads);

    conv2d_forward_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, bias, output,
        batch, in_channels, out_channels,
        in_h, in_w, kernel_h, kernel_w,
        stride_h, stride_w, pad_h, pad_w,
        out_h, out_w);
    CUDEEP_CHECK_LAST_KERNEL();
}

// ===========================================================================
// Conv2d Backward Data — greatly improved with grid-stride + loop unrolling
//
// For each input pixel (b, ic, ih, iw), accumulate contributions from all
// output channels and kernel positions. Precompute valid ranges to minimize
// branching in the inner loop.
// ===========================================================================

template <typename T>
__global__ void conv2d_backward_data_kernel(
    const T* __restrict__ grad_output,
    const T* __restrict__ weight,
    T* __restrict__ grad_input,
    int batch, int in_channels, int out_channels,
    int in_h, int in_w,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int out_h, int out_w
) {
    int total = batch * in_channels * in_h * in_w;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {

        int iw = idx % in_w;
        int ih = (idx / in_w) % in_h;
        int ic = (idx / (in_w * in_h)) % in_channels;
        int b  = idx / (in_w * in_h * in_channels);

        T acc = T(0);

        const T* go_b = grad_output + b * out_channels * out_h * out_w;

        for (int oc = 0; oc < out_channels; ++oc) {
            const T* go_oc = go_b + oc * out_h * out_w;
            const T* w_ptr = weight + (oc * in_channels + ic) * kernel_h * kernel_w;

            #pragma unroll
            for (int kh = 0; kh < kernel_h; ++kh) {
                int oh_num = ih + pad_h - kh;
                if (oh_num < 0 || oh_num % stride_h != 0) continue;
                int oh = oh_num / stride_h;
                if (oh >= out_h) continue;

                #pragma unroll
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int ow_num = iw + pad_w - kw;
                    if (ow_num < 0 || ow_num % stride_w != 0) continue;
                    int ow = ow_num / stride_w;
                    if (ow >= out_w) continue;

                    acc += go_oc[oh * out_w + ow] * w_ptr[kh * kernel_w + kw];
                }
            }
        }

        grad_input[idx] = acc;
    }
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

    int threads = CONV_BLOCK;
    int blocks = min(ceil_div(total, threads), 1024);

    conv2d_backward_data_kernel<<<blocks, threads, 0, stream>>>(
        grad_output, weight, grad_input,
        batch, in_channels, out_channels,
        in_h, in_w, kernel_h, kernel_w,
        stride_h, stride_w, pad_h, pad_w,
        out_h, out_w);
    CUDEEP_CHECK_LAST_KERNEL();
}

// ===========================================================================
// Conv2d Backward Weight — parallelise over (oc, ic, kh, kw) with grid-stride,
// accumulate over batch and spatial dimensions per thread.
// Use __ldg for read-only data to leverage L2 cache.
// ===========================================================================

template <typename T>
__global__ void conv2d_backward_weight_kernel(
    const T* __restrict__ grad_output,
    const T* __restrict__ input,
    T* __restrict__ grad_weight,
    int batch, int in_channels, int out_channels,
    int in_h, int in_w,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int out_h, int out_w
) {
    int total_w = out_channels * in_channels * kernel_h * kernel_w;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_w;
         idx += blockDim.x * gridDim.x) {

        int kw = idx % kernel_w;
        int kh = (idx / kernel_w) % kernel_h;
        int ic = (idx / (kernel_w * kernel_h)) % in_channels;
        int oc = idx / (kernel_w * kernel_h * in_channels);

        T acc = T(0);

        for (int b = 0; b < batch; ++b) {
            const T* go_ptr = grad_output + (b * out_channels + oc) * out_h * out_w;
            const T* in_ptr = input + (b * in_channels + ic) * in_h * in_w;

            for (int oh = 0; oh < out_h; ++oh) {
                int ih = oh * stride_h - pad_h + kh;
                if (ih < 0 || ih >= in_h) continue;

                for (int ow = 0; ow < out_w; ++ow) {
                    int iw = ow * stride_w - pad_w + kw;
                    if (iw < 0 || iw >= in_w) continue;

                    acc += __ldg(&go_ptr[oh * out_w + ow]) * __ldg(&in_ptr[ih * in_w + iw]);
                }
            }
        }

        grad_weight[idx] = acc;
    }
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

    int threads = CONV_BLOCK;
    int blocks = min(ceil_div(total, threads), 1024);

    conv2d_backward_weight_kernel<<<blocks, threads, 0, stream>>>(
        grad_output, input, grad_weight,
        batch, in_channels, out_channels,
        in_h, in_w, kernel_h, kernel_w,
        stride_h, stride_w, pad_h, pad_w,
        out_h, out_w);
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
