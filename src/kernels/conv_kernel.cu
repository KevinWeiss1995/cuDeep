#include "cudeep/kernels/conv.cuh"
#include "cudeep/kernels/matmul.cuh"
#include "cudeep/ptx_intrinsics.cuh"
#include "cudeep/error.cuh"

namespace cudeep {
namespace kernels {

constexpr int CONV_BLOCK = 256;

template <typename T>
void launch_conv2d_winograd_f43(
    const T* input, const T* weight, const T* bias, T* output,
    int batch, int IC, int OC,
    int IH, int IW, int PH, int PW,
    cudaStream_t stream);

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
// Implicit GEMM Conv2d — computes im2col indices on-the-fly
//
// Eliminates the [K_col, B_spatial] intermediate buffer.  Each thread computes
// output[b, oc, oh, ow] by accumulating over K_col = IC*KH*KW, decoding
// the (ic, kh, kw) indices from the linear K index directly.
//
// Falls through to the explicit im2col path for non-3x3 / non-stride-1 cases
// where Winograd doesn't apply.
// ===========================================================================

template <typename T>
__global__ void implicit_gemm_conv2d_kernel(
    const T* __restrict__ input,   // [B, IC, IH, IW]
    const T* __restrict__ weight,  // [OC, IC, KH, KW]
    const T* __restrict__ bias,    // [OC] or nullptr
    T* __restrict__ output,        // [B, OC, OH, OW]
    int batch, int IC, int OC,
    int IH, int IW, int OH, int OW,
    int KH, int KW, int SH, int SW, int PH, int PW,
    int K_col)
{
    int total = batch * OC * OH * OW;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {
        int ow = idx % OW;
        int oh = (idx / OW) % OH;
        int oc = (idx / (OW * OH)) % OC;
        int b  = idx / (OW * OH * OC);

        T acc = bias ? bias[oc] : T(0);

        const T* in_b = input + b * IC * IH * IW;
        const T* w_oc = weight + oc * K_col;

        int ih_base = oh * SH - PH;
        int iw_base = ow * SW - PW;

        for (int ic = 0; ic < IC; ++ic) {
            const T* in_c = in_b + ic * IH * IW;
            const T* w_c  = w_oc + ic * KH * KW;
            for (int kh = 0; kh < KH; ++kh) {
                int ih = ih_base + kh;
                if (ih < 0 || ih >= IH) continue;
                for (int kw = 0; kw < KW; ++kw) {
                    int iw = iw_base + kw;
                    if (iw >= 0 && iw < IW)
                        acc += in_c[ih * IW + iw] * w_c[kh * KW + kw];
                }
            }
        }
        output[idx] = acc;
    }
}

// ===========================================================================
// Conv2d Forward — dispatch hierarchy:
//   1. Winograd F(4,3) for 3x3 stride-1 (above)
//   2. Implicit GEMM for medium-size convolutions (avoids im2col allocation)
//   3. Explicit im2col + GEMM + transpose for very large convolutions
//   4. Naive kernel for tiny convolutions
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

    // Winograd F(4,3) for 3x3 stride-1 convolutions — 4x multiply reduction
    if (kernel_h == 3 && kernel_w == 3 && stride_h == 1 && stride_w == 1 && spatial >= 16) {
        launch_conv2d_winograd_f43<T>(
            input, weight, bias, output,
            batch, in_channels, out_channels,
            in_h, in_w, pad_h, pad_w, stream);
        return;
    }

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

    // Implicit GEMM: avoid im2col allocation for moderate-size problems
    // where the im2col buffer would be large relative to the compute
    size_t col_bytes_est = (size_t)K_col * B_spatial * sizeof(T);
    if (col_bytes_est > 4ULL * 1024 * 1024) {
        int total = batch * out_channels * spatial;
        int blocks = min(ceil_div(total, CONV_BLOCK), 4096);
        implicit_gemm_conv2d_kernel<<<blocks, CONV_BLOCK, 0, stream>>>(
            input, weight, bias, output,
            batch, in_channels, out_channels,
            in_h, in_w, out_h, out_w,
            kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
            K_col);
        CUDEEP_CHECK_LAST_KERNEL();
        return;
    }

    // Explicit im2col + GEMM for small problems where the buffer fits in cache
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
// Conv2d Backward Data — GEMM-based: col2im(weight^T × grad_output_col)
//
// grad_col[K_col, B_spatial] = weight^T[K_col, OC] × grad_out_flat[OC, B_spatial]
// Then scatter col back to grad_input via col2im.
// ===========================================================================

template <typename T>
__global__ void col2im_kernel(
    const T* __restrict__ col,     // [K_col, B_spatial]
    T* __restrict__ grad_input,    // [B, IC, IH, IW]
    int batch, int IC, int IH, int IW,
    int OH, int OW, int KH, int KW,
    int SH, int SW, int PH, int PW,
    int K_col, int spatial, int B_spatial)
{
    int total = batch * IC * IH * IW;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {
        int iw = idx % IW;
        int ih = (idx / IW) % IH;
        int ic = (idx / (IW * IH)) % IC;
        int b  = idx / (IW * IH * IC);

        T acc = T(0);
        for (int kh = 0; kh < KH; ++kh) {
            int oh_num = ih + PH - kh;
            if (oh_num < 0 || oh_num % SH != 0) continue;
            int oh = oh_num / SH;
            if (oh >= OH) continue;
            for (int kw = 0; kw < KW; ++kw) {
                int ow_num = iw + PW - kw;
                if (ow_num < 0 || ow_num % SW != 0) continue;
                int ow = ow_num / SW;
                if (ow >= OW) continue;
                int k = (ic * KH + kh) * KW + kw;
                int s = oh * OW + ow;
                acc += col[k * B_spatial + b * spatial + s];
            }
        }
        grad_input[idx] = acc;
    }
}

// Reshape grad_output [B, OC, OH, OW] → [OC, B_spatial]
template <typename T>
__global__ void nchw_to_oc_bspatial_kernel(
    const T* __restrict__ src,  // [B, OC, spatial]
    T* __restrict__ dst,        // [OC, B_spatial]
    int batch, int OC, int spatial, int B_spatial)
{
    int total = OC * B_spatial;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {
        int bs = idx % B_spatial;
        int oc = idx / B_spatial;
        int b  = bs / spatial;
        int s  = bs % spatial;
        dst[idx] = src[(b * OC + oc) * spatial + s];
    }
}

// Transpose weight [OC, IC*KH*KW] → [IC*KH*KW, OC]
template <typename T>
__global__ void transpose_weight_kernel(
    const T* __restrict__ src, T* __restrict__ dst,
    int rows, int cols)
{
    int total = rows * cols;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {
        int c = idx % cols;
        int r = idx / cols;
        dst[c * rows + r] = src[idx];
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
    int K_col     = in_channels * kernel_h * kernel_w;
    int spatial   = out_h * out_w;
    int B_spatial = batch * spatial;

    // Transpose weight [OC, K_col] → weight_t [K_col, OC]
    T* weight_t = nullptr;
    CUDEEP_CHECK_CUDA(cudaMallocAsync(&weight_t, (size_t)K_col * out_channels * sizeof(T), stream));
    {
        int n = out_channels * K_col;
        transpose_weight_kernel<<<min(ceil_div(n, 256), 4096), 256, 0, stream>>>(
            weight, weight_t, out_channels, K_col);
        CUDEEP_CHECK_LAST_KERNEL();
    }

    // Reshape grad_output to [OC, B_spatial]
    T* go_flat = nullptr;
    CUDEEP_CHECK_CUDA(cudaMallocAsync(&go_flat, (size_t)out_channels * B_spatial * sizeof(T), stream));
    if (batch == 1) {
        CUDEEP_CHECK_CUDA(cudaMemcpyAsync(go_flat, grad_output,
            (size_t)out_channels * B_spatial * sizeof(T),
            cudaMemcpyDeviceToDevice, stream));
    } else {
        int n = out_channels * B_spatial;
        nchw_to_oc_bspatial_kernel<<<min(ceil_div(n, 256), 4096), 256, 0, stream>>>(
            grad_output, go_flat, batch, out_channels, spatial, B_spatial);
        CUDEEP_CHECK_LAST_KERNEL();
    }

    // GEMM: grad_col [K_col, B_spatial] = weight_t [K_col, OC] × go_flat [OC, B_spatial]
    T* grad_col = nullptr;
    CUDEEP_CHECK_CUDA(cudaMallocAsync(&grad_col, (size_t)K_col * B_spatial * sizeof(T), stream));
    launch_matmul_kernel<T>(weight_t, go_flat, grad_col, K_col, B_spatial, out_channels, stream);

    // col2im: scatter grad_col back to grad_input
    CUDEEP_CHECK_CUDA(cudaMemsetAsync(grad_input, 0,
        (size_t)batch * in_channels * in_h * in_w * sizeof(T), stream));
    {
        int n = batch * in_channels * in_h * in_w;
        col2im_kernel<<<min(ceil_div(n, 256), 4096), 256, 0, stream>>>(
            grad_col, grad_input, batch, in_channels, in_h, in_w,
            out_h, out_w, kernel_h, kernel_w,
            stride_h, stride_w, pad_h, pad_w,
            K_col, spatial, B_spatial);
        CUDEEP_CHECK_LAST_KERNEL();
    }

    CUDEEP_CHECK_CUDA(cudaFreeAsync(weight_t, stream));
    CUDEEP_CHECK_CUDA(cudaFreeAsync(go_flat, stream));
    CUDEEP_CHECK_CUDA(cudaFreeAsync(grad_col, stream));
}

// ===========================================================================
// Conv2d Backward Weight — GEMM-based: grad_output_col × input_col^T
//
// grad_weight[OC, K_col] = grad_out_flat[OC, B_spatial] × col^T[B_spatial, K_col]
// ===========================================================================

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
    int K_col     = in_channels * kernel_h * kernel_w;
    int spatial   = out_h * out_w;
    int B_spatial = batch * spatial;

    // Build im2col of input: col [K_col, B_spatial]
    T* col = nullptr;
    CUDEEP_CHECK_CUDA(cudaMallocAsync(&col, (size_t)K_col * B_spatial * sizeof(T), stream));
    {
        int n = batch * K_col * spatial;
        im2col_kernel<<<min(ceil_div(n, 256), 4096), 256, 0, stream>>>(
            input, col, batch, in_channels, in_h, in_w,
            out_h, out_w, kernel_h, kernel_w,
            stride_h, stride_w, pad_h, pad_w,
            K_col, spatial, B_spatial);
        CUDEEP_CHECK_LAST_KERNEL();
    }

    // Reshape grad_output [B, OC, OH, OW] → [OC, B_spatial]
    T* go_flat = nullptr;
    CUDEEP_CHECK_CUDA(cudaMallocAsync(&go_flat, (size_t)out_channels * B_spatial * sizeof(T), stream));
    if (batch == 1) {
        CUDEEP_CHECK_CUDA(cudaMemcpyAsync(go_flat, grad_output,
            (size_t)out_channels * B_spatial * sizeof(T),
            cudaMemcpyDeviceToDevice, stream));
    } else {
        int n = out_channels * B_spatial;
        nchw_to_oc_bspatial_kernel<<<min(ceil_div(n, 256), 4096), 256, 0, stream>>>(
            grad_output, go_flat, batch, out_channels, spatial, B_spatial);
        CUDEEP_CHECK_LAST_KERNEL();
    }

    // Transpose col [K_col, B_spatial] → col_t [B_spatial, K_col]
    T* col_t = nullptr;
    CUDEEP_CHECK_CUDA(cudaMallocAsync(&col_t, (size_t)K_col * B_spatial * sizeof(T), stream));
    {
        int n = K_col * B_spatial;
        transpose_weight_kernel<<<min(ceil_div(n, 256), 4096), 256, 0, stream>>>(
            col, col_t, K_col, B_spatial);
        CUDEEP_CHECK_LAST_KERNEL();
    }

    // GEMM: grad_weight [OC, K_col] = go_flat [OC, B_spatial] × col_t [B_spatial, K_col]
    launch_matmul_kernel<T>(go_flat, col_t, grad_weight, out_channels, K_col, B_spatial, stream);

    CUDEEP_CHECK_CUDA(cudaFreeAsync(col, stream));
    CUDEEP_CHECK_CUDA(cudaFreeAsync(go_flat, stream));
    CUDEEP_CHECK_CUDA(cudaFreeAsync(col_t, stream));
}

// ===========================================================================
// Winograd F(4,3) — 3x3 stride-1 convolutions with 4x multiply reduction
//
// Output tile 4x4, input tile 6x6.  36 pointwise multiplies per output tile
// vs 144 for direct convolution (4x reduction).
//
// Layout:  Weights [36][OC][IC], Input [36][IC][tiles], Output [36][OC][tiles]
// The 36 GEMMs along the transform dimension reuse the optimized GEMM kernel.
// ===========================================================================

namespace winograd {

// B^T * d (column transform): d[6] → t[6]
template <typename T>
__device__ __forceinline__ void input_transform_col(
    T d0, T d1, T d2, T d3, T d4, T d5,
    T& t0, T& t1, T& t2, T& t3, T& t4, T& t5)
{
    t0 = T(4)*d0 - T(5)*d2 + d4;
    t1 = -T(4)*d1 - T(4)*d2 + d3 + d4;
    t2 = T(4)*d1 - T(4)*d2 - d3 + d4;
    t3 = -T(2)*d1 - d2 + T(2)*d3 + d4;
    t4 = T(2)*d1 - d2 - T(2)*d3 + d4;
    t5 = T(4)*d1 - T(5)*d3 + d5;
}

// A^T * m (column transform): m[6] → o[4]
template <typename T>
__device__ __forceinline__ void output_transform_col(
    T m0, T m1, T m2, T m3, T m4, T m5,
    T& o0, T& o1, T& o2, T& o3)
{
    o0 = m0 + m1 + m2 + m3 + m4;
    o1 = m1 - m2 + T(2)*m3 - T(2)*m4;
    o2 = m1 + m2 + T(4)*m3 + T(4)*m4;
    o3 = m1 - m2 + T(8)*m3 - T(8)*m4 + m5;
}

// G * g (column transform): g[3] → u[6]
template <typename T>
__device__ __forceinline__ void weight_transform_col(
    T g0, T g1, T g2,
    T& u0, T& u1, T& u2, T& u3, T& u4, T& u5)
{
    constexpr T c4  = T(1)/T(4);
    constexpr T c6  = T(1)/T(6);
    constexpr T c12 = T(1)/T(12);
    constexpr T c24 = T(1)/T(24);
    u0 = c4 * g0;
    u1 = -c6 * (g0 + g1 + g2);
    u2 = -c6 * (g0 - g1 + g2);
    u3 = c24 * g0 + c12 * g1 + c6 * g2;
    u4 = c24 * g0 - c12 * g1 + c6 * g2;
    u5 = g2;
}

// Transform 3x3 weight → 6x6 Winograd domain
// One thread per (oc, ic) pair
template <typename T>
__global__ void weight_transform_kernel(
    const T* __restrict__ weight,  // [OC, IC, 3, 3]
    T* __restrict__ U,             // [36, OC, IC]
    int OC, int IC)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= OC * IC) return;
    int oc = idx / IC;
    int ic = idx % IC;

    const T* g = weight + (oc * IC + ic) * 9;

    T tmp[6][3];
    for (int j = 0; j < 3; ++j)
        weight_transform_col(g[0*3+j], g[1*3+j], g[2*3+j],
                             tmp[0][j], tmp[1][j], tmp[2][j],
                             tmp[3][j], tmp[4][j], tmp[5][j]);

    for (int i = 0; i < 6; ++i) {
        T u[6];
        weight_transform_col(tmp[i][0], tmp[i][1], tmp[i][2],
                             u[0], u[1], u[2], u[3], u[4], u[5]);
        for (int j = 0; j < 6; ++j)
            U[(i * 6 + j) * OC * IC + oc * IC + ic] = u[j];
    }
}

// Transform input tiles to Winograd domain
// One thread per (ic, tile) pair
template <typename T>
__global__ void input_transform_kernel(
    const T* __restrict__ input,  // [B, IC, IH, IW]
    T* __restrict__ V,            // [36, IC, total_tiles]
    int batch, int IC, int IH, int IW,
    int tiles_h, int tiles_w, int total_tiles,
    int PH, int PW)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= IC * total_tiles) return;
    int ic   = idx / total_tiles;
    int tile = idx % total_tiles;

    int b   = tile / (tiles_h * tiles_w);
    int rem = tile % (tiles_h * tiles_w);
    int th  = rem / tiles_w;
    int tw  = rem % tiles_w;

    int h_start = th * 4 - PH;
    int w_start = tw * 4 - PW;

    T d[6][6];
    const T* in_ptr = input + (b * IC + ic) * IH * IW;
    for (int i = 0; i < 6; ++i) {
        int ih = h_start + i;
        for (int j = 0; j < 6; ++j) {
            int iw = w_start + j;
            d[i][j] = (ih >= 0 && ih < IH && iw >= 0 && iw < IW)
                     ? in_ptr[ih * IW + iw] : T(0);
        }
    }

    T tmp[6][6];
    for (int j = 0; j < 6; ++j)
        input_transform_col(d[0][j], d[1][j], d[2][j],
                            d[3][j], d[4][j], d[5][j],
                            tmp[0][j], tmp[1][j], tmp[2][j],
                            tmp[3][j], tmp[4][j], tmp[5][j]);

    for (int i = 0; i < 6; ++i) {
        T v[6];
        input_transform_col(tmp[i][0], tmp[i][1], tmp[i][2],
                            tmp[i][3], tmp[i][4], tmp[i][5],
                            v[0], v[1], v[2], v[3], v[4], v[5]);
        for (int j = 0; j < 6; ++j)
            V[(i * 6 + j) * IC * total_tiles + ic * total_tiles + tile] = v[j];
    }
}

// Inverse-transform output tiles from Winograd domain + add bias
template <typename T>
__global__ void output_transform_kernel(
    const T* __restrict__ M,      // [36, OC, total_tiles]
    T* __restrict__ output,        // [B, OC, OH, OW]
    const T* __restrict__ bias,    // [OC] or nullptr
    int OC, int OH, int OW,
    int tiles_h, int tiles_w, int total_tiles,
    int batch)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= OC * total_tiles) return;
    int oc   = idx / total_tiles;
    int tile = idx % total_tiles;

    int b   = tile / (tiles_h * tiles_w);
    int rem = tile % (tiles_h * tiles_w);
    int th  = rem / tiles_w;
    int tw  = rem % tiles_w;

    T m[6][6];
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
            m[i][j] = M[(i * 6 + j) * OC * total_tiles + oc * total_tiles + tile];

    T tmp[4][6];
    for (int j = 0; j < 6; ++j)
        output_transform_col(m[0][j], m[1][j], m[2][j],
                             m[3][j], m[4][j], m[5][j],
                             tmp[0][j], tmp[1][j], tmp[2][j], tmp[3][j]);

    T bias_val = bias ? bias[oc] : T(0);
    T* out_ptr = output + (b * OC + oc) * OH * OW;

    for (int i = 0; i < 4; ++i) {
        T o[4];
        output_transform_col(tmp[i][0], tmp[i][1], tmp[i][2],
                             tmp[i][3], tmp[i][4], tmp[i][5],
                             o[0], o[1], o[2], o[3]);
        int oh = th * 4 + i;
        if (oh >= OH) break;
        for (int j = 0; j < 4; ++j) {
            int ow = tw * 4 + j;
            if (ow < OW)
                out_ptr[oh * OW + ow] = o[j] + bias_val;
        }
    }
}

}  // namespace winograd

template <typename T>
void launch_conv2d_winograd_f43(
    const T* input, const T* weight, const T* bias, T* output,
    int batch, int IC, int OC,
    int IH, int IW, int PH, int PW,
    cudaStream_t stream)
{
    int OH = IH + 2 * PH - 2;  // stride=1, kernel=3
    int OW = IW + 2 * PW - 2;
    int tiles_h = (OH + 3) / 4;
    int tiles_w = (OW + 3) / 4;
    int total_tiles = batch * tiles_h * tiles_w;

    T *U = nullptr, *V = nullptr, *M = nullptr;
    size_t U_bytes = 36ULL * OC * IC * sizeof(T);
    size_t V_bytes = 36ULL * IC * total_tiles * sizeof(T);
    size_t M_bytes = 36ULL * OC * total_tiles * sizeof(T);
    CUDEEP_CHECK_CUDA(cudaMallocAsync(&U, U_bytes, stream));
    CUDEEP_CHECK_CUDA(cudaMallocAsync(&V, V_bytes, stream));
    CUDEEP_CHECK_CUDA(cudaMallocAsync(&M, M_bytes, stream));

    {
        int n = OC * IC;
        winograd::weight_transform_kernel<<<ceil_div(n, 256), 256, 0, stream>>>(
            weight, U, OC, IC);
        CUDEEP_CHECK_LAST_KERNEL();
    }

    {
        int n = IC * total_tiles;
        winograd::input_transform_kernel<<<ceil_div(n, 256), 256, 0, stream>>>(
            input, V, batch, IC, IH, IW,
            tiles_h, tiles_w, total_tiles, PH, PW);
        CUDEEP_CHECK_LAST_KERNEL();
    }

    for (int alpha = 0; alpha < 36; ++alpha) {
        const T* U_a = U + (size_t)alpha * OC * IC;
        const T* V_a = V + (size_t)alpha * IC * total_tiles;
        T* M_a       = M + (size_t)alpha * OC * total_tiles;
        launch_matmul_kernel<T>(U_a, V_a, M_a, OC, total_tiles, IC, stream);
    }

    {
        int n = OC * total_tiles;
        winograd::output_transform_kernel<<<ceil_div(n, 256), 256, 0, stream>>>(
            M, output, bias, OC, OH, OW,
            tiles_h, tiles_w, total_tiles, batch);
        CUDEEP_CHECK_LAST_KERNEL();
    }

    CUDEEP_CHECK_CUDA(cudaFreeAsync(U, stream));
    CUDEEP_CHECK_CUDA(cudaFreeAsync(V, stream));
    CUDEEP_CHECK_CUDA(cudaFreeAsync(M, stream));
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
