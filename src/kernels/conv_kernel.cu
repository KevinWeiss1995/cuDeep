#include "cudeep/kernels/conv.cuh"
#include "cudeep/kernels/matmul.cuh"
#include "cudeep/error.cuh"

namespace cudeep {
namespace kernels {

constexpr int CONV_BLOCK = 256;

// ===========================================================================
// im2col — builds [K_col, B*spatial] matrix for a single fused GEMM
//
//   K_col     = IC * KH * KW
//   B_spatial = batch * OH * OW
//
//   col[k * B_spatial + b * spatial + s] = input[b, ic, ih, iw]
//   where k = ic*KH*KW+kh*KW+kw, s = oh*OW+ow, spatial = OH*OW
//
// The row-major layout [K_col, B_spatial] enables:
//   weight[OC, K_col] × col[K_col, B_spatial] = out_flat[OC, B_spatial]
// in a single GEMM call for all batch elements.
// ===========================================================================

template <typename T>
__global__ void im2col_kernel(
    const T* __restrict__ input,
    T* __restrict__ col,
    int batch, int IC, int IH, int IW,
    int OH, int OW, int KH, int KW,
    int SH, int SW, int PH, int PW,
    int K_col, int spatial, int B_spatial
) {
    int total = batch * K_col * spatial;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {
        int s   = idx % spatial;
        int rem = idx / spatial;
        int k   = rem % K_col;
        int b   = rem / K_col;

        int oh = s / OW;
        int ow = s % OW;
        int kw = k % KW;
        int kh = (k / KW) % KH;
        int ic = k / (KW * KH);

        int ih = oh * SH - PH + kh;
        int iw = ow * SW - PW + kw;

        T val = T(0);
        if (ih >= 0 && ih < IH && iw >= 0 && iw < IW)
            val = input[((b * IC + ic) * IH + ih) * IW + iw];

        col[k * B_spatial + b * spatial + s] = val;
    }
}

// ===========================================================================
// Fused transpose + bias:  [OC, B_spatial] → [B, OC, spatial]  (NCHW)
//
// Reads are sequential along B_spatial (stride-1), writes are sequential
// along spatial (stride-1), so both directions are coalesced.
// ===========================================================================

template <typename T>
__global__ void transpose_bias_kernel(
    const T* __restrict__ src,    // [OC, B_spatial]
    T* __restrict__ dst,           // [B, OC, spatial]
    const T* __restrict__ bias,    // [OC] or nullptr
    int OC, int batch, int spatial, int B_spatial, int total
) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {
        int s  = idx % spatial;
        int rem = idx / spatial;
        int b  = rem % batch;
        int oc = rem / batch;

        T val = src[oc * B_spatial + b * spatial + s];
        if (bias) val += bias[oc];
        dst[b * OC * spatial + oc * spatial + s] = val;
    }
}

// No-bias, no-transpose path for batch==1 (output layout matches directly)
template <typename T>
__global__ void bias_add_kernel(
    T* __restrict__ output, const T* __restrict__ bias,
    int OC, int spatial, int total
) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {
        int oc = (idx / spatial) % OC;
        output[idx] += bias[oc];
    }
}

// ===========================================================================
// Naive conv2d forward — fallback for tiny convolutions
// ===========================================================================

template <typename T>
__global__ void conv2d_forward_naive_kernel(
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
    int total = batch * out_channels * out_h * out_w;
    if (idx >= total) return;

    int ow = idx % out_w;
    int oh = (idx / out_w) % out_h;
    int oc = (idx / (out_h * out_w)) % out_channels;
    int b  = idx / (out_h * out_w * out_channels);

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

// ===========================================================================
// Conv2d Forward — im2col + single fused GEMM + transpose
//
// Strategy:
//   1. im2col builds col[K_col, B_spatial] for ALL batches at once
//   2. One GEMM: weight[OC, K_col] × col[K_col, B_spatial] → flat[OC, B_spatial]
//      When B_spatial is a multiple of 128, the aligned TC kernel fires.
//   3. transpose_bias converts [OC, B_spatial] → [B, OC, spatial] (NCHW)
//      and fuses the bias addition in the same memory pass.
//
//   For batch==1 the GEMM output is already NCHW, so we skip the transpose.
// ===========================================================================

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
    int K_col     = in_channels * kernel_h * kernel_w;
    int spatial   = out_h * out_w;
    int B_spatial = batch * spatial;

    if (K_col < 9 || spatial < 4) {
        int total = batch * out_channels * spatial;
        int blocks = ceil_div(total, CONV_BLOCK);
        conv2d_forward_naive_kernel<<<blocks, CONV_BLOCK, 0, stream>>>(
            input, weight, bias, output,
            batch, in_channels, out_channels,
            in_h, in_w, kernel_h, kernel_w,
            stride_h, stride_w, pad_h, pad_w,
            out_h, out_w);
        CUDEEP_CHECK_LAST_KERNEL();
        return;
    }

    // Allocate im2col workspace: [K_col, B_spatial]
    T* col = nullptr;
    size_t col_bytes = (size_t)K_col * B_spatial * sizeof(T);
    CUDEEP_CHECK_CUDA(cudaMallocAsync(&col, col_bytes, stream));

    // Step 1: im2col
    {
        int total = batch * K_col * spatial;
        int blocks = min(ceil_div(total, CONV_BLOCK), 4096);
        im2col_kernel<<<blocks, CONV_BLOCK, 0, stream>>>(
            input, col, batch, in_channels, in_h, in_w,
            out_h, out_w, kernel_h, kernel_w,
            stride_h, stride_w, pad_h, pad_w,
            K_col, spatial, B_spatial);
        CUDEEP_CHECK_LAST_KERNEL();
    }

    if (batch == 1) {
        // GEMM output [OC, spatial] IS NCHW for B=1, write directly
        launch_matmul_kernel<T>(
            weight, col, output,
            out_channels, B_spatial, K_col,
            stream);

        if (bias) {
            int total = out_channels * spatial;
            int blocks = min(ceil_div(total, CONV_BLOCK), 4096);
            bias_add_kernel<<<blocks, CONV_BLOCK, 0, stream>>>(
                output, bias, out_channels, spatial, total);
            CUDEEP_CHECK_LAST_KERNEL();
        }
    } else {
        // Step 2: single GEMM → out_flat[OC, B_spatial]
        T* out_flat = nullptr;
        size_t out_bytes = (size_t)out_channels * B_spatial * sizeof(T);
        CUDEEP_CHECK_CUDA(cudaMallocAsync(&out_flat, out_bytes, stream));

        launch_matmul_kernel<T>(
            weight, col, out_flat,
            out_channels, B_spatial, K_col,
            stream);

        // Step 3: transpose [OC, B_spatial] → [B, OC, spatial] + bias
        {
            int total = out_channels * B_spatial;
            int blocks = min(ceil_div(total, CONV_BLOCK), 4096);
            transpose_bias_kernel<<<blocks, CONV_BLOCK, 0, stream>>>(
                out_flat, output, bias,
                out_channels, batch, spatial, B_spatial, total);
            CUDEEP_CHECK_LAST_KERNEL();
        }

        CUDEEP_CHECK_CUDA(cudaFreeAsync(out_flat, stream));
    }

    CUDEEP_CHECK_CUDA(cudaFreeAsync(col, stream));
}

// ===========================================================================
// Conv2d Backward Data (unchanged — future: col2im + GEMM)
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
    int blocks = min(ceil_div(total, CONV_BLOCK), 1024);
    conv2d_backward_data_kernel<<<blocks, CONV_BLOCK, 0, stream>>>(
        grad_output, weight, grad_input,
        batch, in_channels, out_channels,
        in_h, in_w, kernel_h, kernel_w,
        stride_h, stride_w, pad_h, pad_w,
        out_h, out_w);
    CUDEEP_CHECK_LAST_KERNEL();
}

// ===========================================================================
// Conv2d Backward Weight (unchanged — future: GEMM-based)
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
    int blocks = min(ceil_div(total, CONV_BLOCK), 1024);
    conv2d_backward_weight_kernel<<<blocks, CONV_BLOCK, 0, stream>>>(
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
