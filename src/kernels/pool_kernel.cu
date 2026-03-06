#include "cudeep/kernels/pool.cuh"
#include "cudeep/ptx_intrinsics.cuh"
#include "cudeep/error.cuh"

namespace cudeep {
namespace kernels {

// ---- MaxPool2d forward ----
// Vectorized variant: processes 4 adjacent output-width elements per thread
// when out_w >= 4, using ldg_v4 to load aligned input tile rows.

template <typename T>
__global__ void maxpool2d_forward_vec4_kernel(
    const T* __restrict__ input, T* __restrict__ output,
    int batch, int channels,
    int in_h, int in_w,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int out_h, int out_w
) {
    constexpr int VW = 4;
    int total_vec = batch * channels * out_h * (out_w / VW);
    for (int vid = blockIdx.x * blockDim.x + threadIdx.x;
         vid < total_vec;
         vid += blockDim.x * gridDim.x) {
        int ow4 = vid % (out_w / VW);
        int oh  = (vid / (out_w / VW)) % out_h;
        int c   = (vid / ((out_w / VW) * out_h)) % channels;
        int b   = vid / ((out_w / VW) * out_h * channels);

        T mx[VW];
        #pragma unroll
        for (int j = 0; j < VW; ++j) mx[j] = T(-1e38);

        const T* in_bc = input + ((b * channels + c) * in_h) * in_w;
        for (int kh = 0; kh < kernel_h; ++kh) {
            int ih = oh * stride_h - pad_h + kh;
            if (ih < 0 || ih >= in_h) continue;
            const T* row = in_bc + ih * in_w;
            for (int kw = 0; kw < kernel_w; ++kw) {
                #pragma unroll
                for (int j = 0; j < VW; ++j) {
                    int iw = (ow4 * VW + j) * stride_w - pad_w + kw;
                    if (iw >= 0 && iw < in_w)
                        mx[j] = max(mx[j], row[iw]);
                }
            }
        }

        int out_base = ((b * channels + c) * out_h + oh) * out_w + ow4 * VW;
        if constexpr (sizeof(T) == 4) {
            float4 vo = make_float4(mx[0], mx[1], mx[2], mx[3]);
            Vec4<T>::store(&output[out_base], vo);
        } else {
            #pragma unroll
            for (int j = 0; j < VW; ++j) output[out_base + j] = mx[j];
        }
    }

    // Scalar tail for out_w not divisible by 4
    int tail_start = (out_w / VW) * VW;
    int total_tail = batch * channels * out_h * (out_w - tail_start);
    for (int tid_t = blockIdx.x * blockDim.x + threadIdx.x;
         tid_t < total_tail;
         tid_t += blockDim.x * gridDim.x) {
        int ow = tail_start + tid_t % (out_w - tail_start);
        int oh = (tid_t / (out_w - tail_start)) % out_h;
        int c  = (tid_t / ((out_w - tail_start) * out_h)) % channels;
        int b  = tid_t / ((out_w - tail_start) * out_h * channels);

        T max_val = T(-1e38);
        for (int kh = 0; kh < kernel_h; ++kh) {
            int ih = oh * stride_h - pad_h + kh;
            if (ih < 0 || ih >= in_h) continue;
            for (int kw = 0; kw < kernel_w; ++kw) {
                int iw = ow * stride_w - pad_w + kw;
                if (iw >= 0 && iw < in_w) {
                    int in_idx = ((b * channels + c) * in_h + ih) * in_w + iw;
                    max_val = max(max_val, input[in_idx]);
                }
            }
        }
        int idx = ((b * channels + c) * out_h + oh) * out_w + ow;
        output[idx] = max_val;
    }
}

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

    if (out_w >= 4 && sizeof(T) == 4) {
        int vec_total = batch * channels * out_h * (out_w / 4);
        int blocks = min(ceil_div(vec_total, threads), 4096);
        maxpool2d_forward_vec4_kernel<<<blocks, threads, 0, stream>>>(
            input, output, batch, channels, in_h, in_w,
            kernel_h, kernel_w, stride_h, stride_w,
            pad_h, pad_w, out_h, out_w);
    } else {
        int blocks = ceil_div(total, threads);
        maxpool2d_forward_kernel<<<blocks, threads, 0, stream>>>(
            input, output, batch, channels, in_h, in_w,
            kernel_h, kernel_w, stride_h, stride_w,
            pad_h, pad_w, out_h, out_w);
    }
    CUDEEP_CHECK_LAST_KERNEL();
}

// ---- AvgPool2d forward ----

template <typename T>
__global__ void avgpool2d_forward_vec4_kernel(
    const T* __restrict__ input, T* __restrict__ output,
    int batch, int channels,
    int in_h, int in_w,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int out_h, int out_w
) {
    constexpr int VW = 4;
    int total_vec = batch * channels * out_h * (out_w / VW);
    for (int vid = blockIdx.x * blockDim.x + threadIdx.x;
         vid < total_vec;
         vid += blockDim.x * gridDim.x) {
        int ow4 = vid % (out_w / VW);
        int oh  = (vid / (out_w / VW)) % out_h;
        int c   = (vid / ((out_w / VW) * out_h)) % channels;
        int b   = vid / ((out_w / VW) * out_h * channels);

        T sm[VW] = {T(0), T(0), T(0), T(0)};
        int cnt[VW] = {0, 0, 0, 0};

        const T* in_bc = input + ((b * channels + c) * in_h) * in_w;
        for (int kh = 0; kh < kernel_h; ++kh) {
            int ih = oh * stride_h - pad_h + kh;
            if (ih < 0 || ih >= in_h) continue;
            const T* row = in_bc + ih * in_w;
            for (int kw = 0; kw < kernel_w; ++kw) {
                #pragma unroll
                for (int j = 0; j < VW; ++j) {
                    int iw = (ow4 * VW + j) * stride_w - pad_w + kw;
                    if (iw >= 0 && iw < in_w) {
                        sm[j] += row[iw];
                        cnt[j]++;
                    }
                }
            }
        }

        int out_base = ((b * channels + c) * out_h + oh) * out_w + ow4 * VW;
        if constexpr (sizeof(T) == 4) {
            float4 vo;
            vo.x = cnt[0] > 0 ? sm[0] / T(cnt[0]) : T(0);
            vo.y = cnt[1] > 0 ? sm[1] / T(cnt[1]) : T(0);
            vo.z = cnt[2] > 0 ? sm[2] / T(cnt[2]) : T(0);
            vo.w = cnt[3] > 0 ? sm[3] / T(cnt[3]) : T(0);
            Vec4<T>::store(&output[out_base], vo);
        } else {
            #pragma unroll
            for (int j = 0; j < VW; ++j)
                output[out_base + j] = cnt[j] > 0 ? sm[j] / T(cnt[j]) : T(0);
        }
    }

    int tail_start = (out_w / VW) * VW;
    int total_tail = batch * channels * out_h * (out_w - tail_start);
    for (int tid_t = blockIdx.x * blockDim.x + threadIdx.x;
         tid_t < total_tail;
         tid_t += blockDim.x * gridDim.x) {
        int ow = tail_start + tid_t % (out_w - tail_start);
        int oh = (tid_t / (out_w - tail_start)) % out_h;
        int c  = (tid_t / ((out_w - tail_start) * out_h)) % channels;
        int b  = tid_t / ((out_w - tail_start) * out_h * channels);

        T sum = T(0);
        int count = 0;
        for (int kh = 0; kh < kernel_h; ++kh) {
            int ih = oh * stride_h - pad_h + kh;
            if (ih < 0 || ih >= in_h) continue;
            for (int kw = 0; kw < kernel_w; ++kw) {
                int iw = ow * stride_w - pad_w + kw;
                if (iw >= 0 && iw < in_w) {
                    sum += input[((b * channels + c) * in_h + ih) * in_w + iw];
                    ++count;
                }
            }
        }
        int idx = ((b * channels + c) * out_h + oh) * out_w + ow;
        output[idx] = count > 0 ? sum / T(count) : T(0);
    }
}

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

    if (out_w >= 4 && sizeof(T) == 4) {
        int vec_total = batch * channels * out_h * (out_w / 4);
        int blocks = min(ceil_div(vec_total, threads), 4096);
        avgpool2d_forward_vec4_kernel<<<blocks, threads, 0, stream>>>(
            input, output, batch, channels, in_h, in_w,
            kernel_h, kernel_w, stride_h, stride_w,
            pad_h, pad_w, out_h, out_w);
    } else {
        int blocks = ceil_div(total, threads);
        avgpool2d_forward_kernel<<<blocks, threads, 0, stream>>>(
            input, output, batch, channels, in_h, in_w,
            kernel_h, kernel_w, stride_h, stride_w,
            pad_h, pad_w, out_h, out_w);
    }
    CUDEEP_CHECK_LAST_KERNEL();
}

// ---- Explicit instantiations ----

template void launch_maxpool2d_forward_kernel<float>(const float*, float*, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
template void launch_maxpool2d_forward_kernel<double>(const double*, double*, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
template void launch_avgpool2d_forward_kernel<float>(const float*, float*, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
template void launch_avgpool2d_forward_kernel<double>(const double*, double*, int, int, int, int, int, int, int, int, int, int, cudaStream_t);

}  // namespace kernels
}  // namespace cudeep
