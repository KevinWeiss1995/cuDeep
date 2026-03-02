/*
 * bench_vs_reference.cu — cuDeep vs cuBLAS + cuDNN head-to-head benchmark
 *
 * Compares every major cuDeep kernel category against the authoritative
 * NVIDIA library implementation (cuBLAS for GEMM, cuDNN for everything else).
 *
 * Build:  (handled by CMake, or manually)
 *   nvcc -O3 -std=c++17 -I../include bench_vs_reference.cu \
 *        -L../build/lib -lcudeep -lcublas -lcudnn -o bench_vs_reference
 *
 * Run:
 *   LD_LIBRARY_PATH=../build/lib:$LD_LIBRARY_PATH ./bench_vs_reference
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <functional>

#include "cudeep/kernels/matmul.cuh"
#include "cudeep/kernels/conv.cuh"
#include "cudeep/kernels/activation.cuh"
#include "cudeep/kernels/pool.cuh"
#include "cudeep/kernels/norm.cuh"
#include "cudeep/kernels/reduce.cuh"
#include "cudeep/kernels/loss.cuh"
#include "cudeep/error.cuh"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#define CHECK_CUDA(call) do {                                          \
    cudaError_t err = (call);                                          \
    if (err != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err));                               \
        exit(1);                                                       \
    }                                                                  \
} while(0)

#define CHECK_CUBLAS(call) do {                                        \
    cublasStatus_t s = (call);                                         \
    if (s != CUBLAS_STATUS_SUCCESS) {                                  \
        fprintf(stderr, "cuBLAS error %s:%d: %d\n",                    \
                __FILE__, __LINE__, (int)s);                           \
        exit(1);                                                       \
    }                                                                  \
} while(0)

#define CHECK_CUDNN(call) do {                                         \
    cudnnStatus_t s = (call);                                          \
    if (s != CUDNN_STATUS_SUCCESS) {                                   \
        fprintf(stderr, "cuDNN error %s:%d: %s\n", __FILE__, __LINE__, \
                cudnnGetErrorString(s));                                \
        exit(1);                                                       \
    }                                                                  \
} while(0)

static void fill_random(float* d, size_t n) {
    std::vector<float> h(n);
    for (size_t i = 0; i < n; ++i) h[i] = (float)rand() / RAND_MAX - 0.5f;
    CHECK_CUDA(cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
}

static void fill_zero(float* d, size_t n) {
    CHECK_CUDA(cudaMemset(d, 0, n * sizeof(float)));
}

static void fill_ones(float* d, size_t n) {
    std::vector<float> h(n, 1.0f);
    CHECK_CUDA(cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
}

struct BenchResult {
    std::string name;
    std::string config;
    float cudeep_ms;
    float ref_ms;
    double cudeep_gflops;
    double ref_gflops;

    double speedup() const { return ref_ms > 0 ? ref_ms / cudeep_ms : 0; }
    bool is_win()  const { return cudeep_ms < ref_ms * 0.97; }
    bool is_loss() const { return cudeep_ms > ref_ms * 1.03; }
    bool is_tie()  const { return !is_win() && !is_loss(); }
};

using BenchFn = std::function<void()>;

// ANSI color codes
#define C_RESET   "\033[0m"
#define C_GREEN   "\033[1;32m"
#define C_RED     "\033[1;31m"
#define C_YELLOW  "\033[1;33m"
#define C_CYAN    "\033[1;36m"
#define C_BOLD    "\033[1m"
#define C_DIM     "\033[2m"

static float timed_run(BenchFn fn, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) fn();
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) fn();
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return ms / iters;
}

static void print_header() {
    printf("  %-34s %9s %9s %9s %9s  %-18s\n",
           "Kernel", "cuDeep", "Ref", "cuDeep", "Ref", "");
    printf("  %-34s %9s %9s %9s %9s  %-18s\n",
           "", "(ms)", "(ms)", "(GF|GB)", "(GF|GB)", "VERDICT");
    printf("  %s\n", std::string(100, '-').c_str());
}

static void print_result(const BenchResult& r) {
    char cudeep_gf[16], ref_gf[16];
    if (r.cudeep_gflops > 0) snprintf(cudeep_gf, sizeof(cudeep_gf), "%9.1f", r.cudeep_gflops);
    else snprintf(cudeep_gf, sizeof(cudeep_gf), "%9s", "-");
    if (r.ref_gflops > 0) snprintf(ref_gf, sizeof(ref_gf), "%9.1f", r.ref_gflops);
    else snprintf(ref_gf, sizeof(ref_gf), "%9s", "-");

    double sp = r.speedup();
    char verdict[64];

    if (r.is_win())
        snprintf(verdict, sizeof(verdict), "%s>>> %.2fx FASTER%s", C_GREEN, sp, C_RESET);
    else if (r.is_loss())
        snprintf(verdict, sizeof(verdict), "%s<<< %.2fx SLOWER%s", C_RED, 1.0/sp, C_RESET);
    else
        snprintf(verdict, sizeof(verdict), "%s=== TIED%s", C_YELLOW, C_RESET);

    printf("  %-34s %9.3f %9.3f %s %s  %s\n",
           r.name.c_str(), r.cudeep_ms, r.ref_ms,
           cudeep_gf, ref_gf, verdict);
}

static void print_section(const char* title, const char* ref_lib,
                           const std::vector<BenchResult>& results) {
    int wins = 0, losses = 0, ties = 0;
    for (auto& r : results) {
        if (r.is_win()) wins++;
        else if (r.is_loss()) losses++;
        else ties++;
    }

    printf("\n%s\n", std::string(104, '=').c_str());
    printf("  %s%s%s  (vs %s%s%s)\n", C_BOLD, title, C_RESET, C_CYAN, ref_lib, C_RESET);
    printf("%s\n", std::string(104, '=').c_str());
    print_header();
    for (auto& r : results) print_result(r);

    printf("  %s\n", std::string(100, '-').c_str());
    printf("  Scorecard:  %s%d WIN%s", C_GREEN, wins, C_RESET);
    printf("  /  %s%d LOSS%s", C_RED, losses, C_RESET);
    printf("  /  %s%d TIE%s\n", C_YELLOW, ties, C_RESET);
}

// ---------------------------------------------------------------------------
// cuBLAS / cuDNN handles (global for convenience)
// ---------------------------------------------------------------------------

static cublasHandle_t g_cublas;
static cudnnHandle_t  g_cudnn;

// ---------------------------------------------------------------------------
// 1. GEMM — cuDeep vs cuBLAS
// ---------------------------------------------------------------------------

static std::vector<BenchResult> bench_gemm() {
    std::vector<BenchResult> results;
    struct Cfg { int M, N, K; };
    Cfg cfgs[] = {
        {256,  256,  256},
        {512,  512,  512},
        {1024, 1024, 1024},
        {2048, 2048, 2048},
        {4096, 4096, 4096},
        {128,  4096, 512},   // tall-skinny
        {4096, 128,  512},   // wide-short
        {1024, 1024, 4096},  // large K
    };

    for (auto& c : cfgs) {
        int M = c.M, N = c.N, K = c.K;
        size_t sA = (size_t)M * K, sB = (size_t)K * N, sC = (size_t)M * N;
        float *A, *B, *C_cu, *C_ref;
        CHECK_CUDA(cudaMalloc(&A,     sA * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&B,     sB * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&C_cu,  sC * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&C_ref, sC * sizeof(float)));
        fill_random(A, sA);
        fill_random(B, sB);

        int warmup = (M <= 512) ? 50 : 20;
        int iters  = (M <= 512) ? 200 : 100;
        double flops = 2.0 * M * N * K;

        float cu_ms = timed_run([&]{ cudeep::kernels::launch_matmul_kernel_fp32<float>(A, B, C_cu, M, N, K, 0); }, warmup, iters);

        float alpha = 1.0f, beta = 0.0f;
        float cb_ms = timed_run([&]{
            cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                        N, M, K, &alpha, B, N, A, K, &beta, C_ref, N);
        }, warmup, iters);

        char tag[64];
        snprintf(tag, sizeof(tag), "%dx%dx%d", M, N, K);

        results.push_back({
            std::string("SGEMM ") + tag, tag,
            cu_ms, cb_ms,
            flops / (cu_ms * 1e6), flops / (cb_ms * 1e6)
        });

        CHECK_CUDA(cudaFree(A));
        CHECK_CUDA(cudaFree(B));
        CHECK_CUDA(cudaFree(C_cu));
        CHECK_CUDA(cudaFree(C_ref));
    }

    return results;
}

// ---------------------------------------------------------------------------
// 2. GEMM TF32 — cuDeep TC kernel vs cuBLAS TF32
// ---------------------------------------------------------------------------

static std::vector<BenchResult> bench_gemm_tf32() {
    std::vector<BenchResult> results;
    int sizes[] = {256, 512, 1024, 2048, 4096};

    cublasSetMathMode(g_cublas, CUBLAS_TF32_TENSOR_OP_MATH);

    for (int s : sizes) {
        size_t sA = (size_t)s * s;
        float *A, *B, *C_cu, *C_ref;
        CHECK_CUDA(cudaMalloc(&A,     sA * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&B,     sA * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&C_cu,  sA * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&C_ref, sA * sizeof(float)));
        fill_random(A, sA);
        fill_random(B, sA);

        int warmup = (s <= 512) ? 50 : 20;
        int iters  = (s <= 512) ? 200 : 100;
        double flops = 2.0 * s * s * s;

        float cu_ms = timed_run([&]{ cudeep::kernels::launch_matmul_kernel<float>(A, B, C_cu, s, s, s, 0); }, warmup, iters);

        float alpha = 1.0f, beta = 0.0f;
        float cb_ms = timed_run([&]{
            cublasGemmEx(g_cublas,
                         CUBLAS_OP_N, CUBLAS_OP_N,
                         s, s, s,
                         &alpha, B, CUDA_R_32F, s,
                                 A, CUDA_R_32F, s,
                         &beta,  C_ref, CUDA_R_32F, s,
                         CUBLAS_COMPUTE_32F_FAST_TF32,
                         CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }, warmup, iters);

        char tag[32];
        snprintf(tag, sizeof(tag), "%dx%d TF32", s, s);

        results.push_back({
            std::string("GEMM ") + tag, tag,
            cu_ms, cb_ms,
            flops / (cu_ms * 1e6), flops / (cb_ms * 1e6)
        });

        CHECK_CUDA(cudaFree(A));
        CHECK_CUDA(cudaFree(B));
        CHECK_CUDA(cudaFree(C_cu));
        CHECK_CUDA(cudaFree(C_ref));
    }

    cublasSetMathMode(g_cublas, CUBLAS_DEFAULT_MATH);
    return results;
}

// ---------------------------------------------------------------------------
// 3. Conv2d Forward — cuDeep vs cuDNN
// ---------------------------------------------------------------------------

static std::vector<BenchResult> bench_conv2d() {
    std::vector<BenchResult> results;
    struct Cfg { int B, IC, OC, H, W, K, S, P; };
    Cfg cfgs[] = {
        {1,   3,   64,  224, 224, 3, 1, 1},  // ResNet first layer
        {1,   3,   64,  224, 224, 7, 2, 3},  // ResNet 7x7 stem
        {8,   64,  128, 56,  56,  3, 1, 1},  // mid-network
        {8,   128, 256, 28,  28,  3, 1, 1},
        {16,  256, 512, 14,  14,  3, 1, 1},
        {4,   32,  64,  32,  32,  3, 1, 1},  // small feature map
        {1,   512, 512, 7,   7,   3, 1, 1},  // ResNet final block
    };

    for (auto& c : cfgs) {
        int OH = (c.H + 2 * c.P - c.K) / c.S + 1;
        int OW = (c.W + 2 * c.P - c.K) / c.S + 1;

        size_t sX = (size_t)c.B * c.IC * c.H * c.W;
        size_t sW = (size_t)c.OC * c.IC * c.K * c.K;
        size_t sY = (size_t)c.B * c.OC * OH * OW;

        float *dX, *dW, *dY_cu, *dY_ref;
        CHECK_CUDA(cudaMalloc(&dX,     sX * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dW,     sW * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dY_cu,  sY * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dY_ref, sY * sizeof(float)));
        fill_random(dX, sX);
        fill_random(dW, sW);

        double flops = 2.0 * c.B * c.OC * OH * OW * c.IC * c.K * c.K;
        int warmup = 20, iters = 100;

        // cuDeep
        float cu_ms = timed_run([&]{
            cudeep::kernels::launch_conv2d_forward_kernel<float>(
                dX, dW, nullptr, dY_cu,
                c.B, c.IC, c.OC, c.H, c.W, c.K, c.K, c.S, c.S, c.P, c.P, 0);
        }, warmup, iters);

        // cuDNN
        cudnnTensorDescriptor_t xDesc, yDesc;
        cudnnFilterDescriptor_t wDesc;
        cudnnConvolutionDescriptor_t convDesc;
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&xDesc));
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&yDesc));
        CHECK_CUDNN(cudnnCreateFilterDescriptor(&wDesc));
        CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

        CHECK_CUDNN(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                                c.B, c.IC, c.H, c.W));
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                                c.B, c.OC, OH, OW));
        CHECK_CUDNN(cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                                c.OC, c.IC, c.K, c.K));
        CHECK_CUDNN(cudnnSetConvolution2dDescriptor(convDesc, c.P, c.P, c.S, c.S, 1, 1,
                                                     CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
        CHECK_CUDNN(cudnnSetConvolutionMathType(convDesc, CUDNN_DEFAULT_MATH));

        // Find fastest algorithm
        int returnedAlgoCount;
        cudnnConvolutionFwdAlgoPerf_t perfResults[8];
        CHECK_CUDNN(cudnnFindConvolutionForwardAlgorithm(
            g_cudnn, xDesc, wDesc, convDesc, yDesc,
            8, &returnedAlgoCount, perfResults));
        cudnnConvolutionFwdAlgo_t bestAlgo = perfResults[0].algo;

        size_t ws_size = 0;
        CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
            g_cudnn, xDesc, wDesc, convDesc, yDesc, bestAlgo, &ws_size));
        void* ws = nullptr;
        if (ws_size > 0) CHECK_CUDA(cudaMalloc(&ws, ws_size));

        float alpha = 1.0f, beta = 0.0f;
        float ref_ms = timed_run([&]{
            cudnnConvolutionForward(g_cudnn, &alpha, xDesc, dX, wDesc, dW,
                                    convDesc, bestAlgo, ws, ws_size,
                                    &beta, yDesc, dY_ref);
        }, warmup, iters);

        char tag[64];
        snprintf(tag, sizeof(tag), "N%d C%d→%d %dx%d k%d", c.B, c.IC, c.OC, c.H, c.W, c.K);

        results.push_back({
            std::string("Conv2d ") + tag, tag,
            cu_ms, ref_ms,
            flops / (cu_ms * 1e6), flops / (ref_ms * 1e6)
        });

        if (ws) CHECK_CUDA(cudaFree(ws));
        CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
        CHECK_CUDNN(cudnnDestroyFilterDescriptor(wDesc));
        CHECK_CUDNN(cudnnDestroyTensorDescriptor(yDesc));
        CHECK_CUDNN(cudnnDestroyTensorDescriptor(xDesc));
        CHECK_CUDA(cudaFree(dX));
        CHECK_CUDA(cudaFree(dW));
        CHECK_CUDA(cudaFree(dY_cu));
        CHECK_CUDA(cudaFree(dY_ref));
    }

    return results;
}

// ---------------------------------------------------------------------------
// 4. Activations — cuDeep vs cuDNN
// ---------------------------------------------------------------------------

static std::vector<BenchResult> bench_activations() {
    std::vector<BenchResult> results;
    int sizes[] = {1 << 20, 4 << 20, 16 << 20};  // 1M, 4M, 16M

    struct ActCfg {
        const char* name;
        cudeep::kernels::ActivationType cu_type;
        cudnnActivationMode_t cudnn_mode;
        float alpha;
    };
    ActCfg acts[] = {
        {"ReLU",    cudeep::kernels::ActivationType::ReLU,    CUDNN_ACTIVATION_RELU,    0.0f},
        {"Sigmoid", cudeep::kernels::ActivationType::Sigmoid, CUDNN_ACTIVATION_SIGMOID, 0.0f},
        {"Tanh",    cudeep::kernels::ActivationType::Tanh,    CUDNN_ACTIVATION_TANH,    0.0f},
        {"ELU",     cudeep::kernels::ActivationType::SiLU,    CUDNN_ACTIVATION_ELU,     1.0f},
    };

    for (int n : sizes) {
        float *dX, *dY_cu, *dY_ref;
        CHECK_CUDA(cudaMalloc(&dX,     n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dY_cu,  n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dY_ref, n * sizeof(float)));
        fill_random(dX, n);

        cudnnTensorDescriptor_t tDesc;
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&tDesc));
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(tDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                                1, 1, 1, n));

        for (auto& a : acts) {
            double bytes = (double)n * sizeof(float) * 2;  // 1 read + 1 write
            int warmup = 50, iters = 200;

            float cu_ms = timed_run([&]{
                cudeep::kernels::launch_activation_forward_kernel<float>(
                    dX, dY_cu, n, a.cu_type, a.alpha, 0);
            }, warmup, iters);

            cudnnActivationDescriptor_t actDesc;
            CHECK_CUDNN(cudnnCreateActivationDescriptor(&actDesc));
            CHECK_CUDNN(cudnnSetActivationDescriptor(actDesc, a.cudnn_mode,
                                                      CUDNN_PROPAGATE_NAN, a.alpha));
            float one = 1.0f, zero = 0.0f;
            float ref_ms = timed_run([&]{
                cudnnActivationForward(g_cudnn, actDesc, &one, tDesc, dX,
                                        &zero, tDesc, dY_ref);
            }, warmup, iters);
            CHECK_CUDNN(cudnnDestroyActivationDescriptor(actDesc));

            char tag[64];
            snprintf(tag, sizeof(tag), "%s %dM", a.name, n >> 20);
            double bw_cu  = bytes / (cu_ms  * 1e6);
            double bw_ref = bytes / (ref_ms * 1e6);

            results.push_back({
                tag, tag, cu_ms, ref_ms, bw_cu, bw_ref
            });
        }

        CHECK_CUDNN(cudnnDestroyTensorDescriptor(tDesc));
        CHECK_CUDA(cudaFree(dX));
        CHECK_CUDA(cudaFree(dY_cu));
        CHECK_CUDA(cudaFree(dY_ref));
    }

    return results;
}

// ---------------------------------------------------------------------------
// 5. Pooling — cuDeep vs cuDNN
// ---------------------------------------------------------------------------

static std::vector<BenchResult> bench_pooling() {
    std::vector<BenchResult> results;
    struct Cfg { int B, C, H, W, K, S, P; cudnnPoolingMode_t mode; const char* name; };
    Cfg cfgs[] = {
        {8,  64,  32, 32, 2, 2, 0, CUDNN_POOLING_MAX, "MaxPool"},
        {16, 128, 16, 16, 2, 2, 0, CUDNN_POOLING_MAX, "MaxPool"},
        {32, 256, 8,  8,  2, 2, 0, CUDNN_POOLING_MAX, "MaxPool"},
        {8,  64,  32, 32, 2, 2, 0, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, "AvgPool"},
        {16, 128, 16, 16, 2, 2, 0, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, "AvgPool"},
        {32, 256, 8,  8,  2, 2, 0, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, "AvgPool"},
    };

    for (auto& c : cfgs) {
        int OH = (c.H + 2 * c.P - c.K) / c.S + 1;
        int OW = (c.W + 2 * c.P - c.K) / c.S + 1;
        size_t sX = (size_t)c.B * c.C * c.H * c.W;
        size_t sY = (size_t)c.B * c.C * OH * OW;

        float *dX, *dY_cu, *dY_ref;
        CHECK_CUDA(cudaMalloc(&dX,     sX * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dY_cu,  sY * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dY_ref, sY * sizeof(float)));
        fill_random(dX, sX);

        int warmup = 30, iters = 150;
        double bytes = ((double)sX + sY) * sizeof(float);

        bool is_max = (c.mode == CUDNN_POOLING_MAX);
        float cu_ms = timed_run([&]{
            if (is_max)
                cudeep::kernels::launch_maxpool2d_forward_kernel<float>(
                    dX, dY_cu, c.B, c.C, c.H, c.W, c.K, c.K, c.S, c.S, c.P, c.P, 0);
            else
                cudeep::kernels::launch_avgpool2d_forward_kernel<float>(
                    dX, dY_cu, c.B, c.C, c.H, c.W, c.K, c.K, c.S, c.S, c.P, c.P, 0);
        }, warmup, iters);

        // cuDNN
        cudnnTensorDescriptor_t xDesc, yDesc;
        cudnnPoolingDescriptor_t poolDesc;
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&xDesc));
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&yDesc));
        CHECK_CUDNN(cudnnCreatePoolingDescriptor(&poolDesc));

        CHECK_CUDNN(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                                c.B, c.C, c.H, c.W));
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                                c.B, c.C, OH, OW));
        CHECK_CUDNN(cudnnSetPooling2dDescriptor(poolDesc, c.mode, CUDNN_PROPAGATE_NAN,
                                                 c.K, c.K, c.P, c.P, c.S, c.S));
        float one = 1.0f, zero = 0.0f;
        float ref_ms = timed_run([&]{
            cudnnPoolingForward(g_cudnn, poolDesc, &one, xDesc, dX, &zero, yDesc, dY_ref);
        }, warmup, iters);

        char tag[64];
        snprintf(tag, sizeof(tag), "%s N%d C%d %dx%d k%d", c.name, c.B, c.C, c.H, c.W, c.K);
        double bw_cu  = bytes / (cu_ms  * 1e6);
        double bw_ref = bytes / (ref_ms * 1e6);

        results.push_back({ tag, tag, cu_ms, ref_ms, bw_cu, bw_ref });

        CHECK_CUDNN(cudnnDestroyPoolingDescriptor(poolDesc));
        CHECK_CUDNN(cudnnDestroyTensorDescriptor(yDesc));
        CHECK_CUDNN(cudnnDestroyTensorDescriptor(xDesc));
        CHECK_CUDA(cudaFree(dX));
        CHECK_CUDA(cudaFree(dY_cu));
        CHECK_CUDA(cudaFree(dY_ref));
    }

    return results;
}

// ---------------------------------------------------------------------------
// 6. BatchNorm — cuDeep vs cuDNN
// ---------------------------------------------------------------------------

static std::vector<BenchResult> bench_batchnorm() {
    std::vector<BenchResult> results;
    struct Cfg { int B, C, H, W; };
    Cfg cfgs[] = {
        {8,  64,  56, 56},
        {16, 128, 28, 28},
        {32, 256, 14, 14},
        {64, 512, 7,  7},
    };

    for (auto& c : cfgs) {
        size_t sX = (size_t)c.B * c.C * c.H * c.W;

        float *dX, *dY_cu, *dY_ref;
        float *dGamma, *dBeta, *dRunMean, *dRunVar, *dSaveMean, *dSaveInvVar;
        CHECK_CUDA(cudaMalloc(&dX,          sX * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dY_cu,       sX * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dY_ref,      sX * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dGamma,      c.C * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dBeta,       c.C * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dRunMean,    c.C * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dRunVar,     c.C * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dSaveMean,   c.C * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dSaveInvVar, c.C * sizeof(float)));
        fill_random(dX, sX);
        fill_ones(dGamma, c.C);
        fill_zero(dBeta, c.C);
        fill_zero(dRunMean, c.C);
        fill_ones(dRunVar, c.C);

        int warmup = 20, iters = 100;
        double bytes = (double)sX * sizeof(float) * 2;

        float cu_ms = timed_run([&]{
            cudeep::kernels::launch_batchnorm_forward_kernel<float>(
                dX, dY_cu, dGamma, dBeta, dRunMean, dRunVar,
                c.B, c.C, c.H * c.W, 1e-5f, 0.1f, true, 0);
        }, warmup, iters);

        // cuDNN
        cudnnTensorDescriptor_t xDesc, bnDesc;
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&xDesc));
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&bnDesc));
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                                c.B, c.C, c.H, c.W));
        CHECK_CUDNN(cudnnDeriveBNTensorDescriptor(bnDesc, xDesc, CUDNN_BATCHNORM_SPATIAL));

        float one = 1.0f, zero = 0.0f;
        double expAvgFactor = 0.1;
        float ref_ms = timed_run([&]{
            cudnnBatchNormalizationForwardTraining(
                g_cudnn, CUDNN_BATCHNORM_SPATIAL,
                &one, &zero, xDesc, dX, xDesc, dY_ref,
                bnDesc, dGamma, dBeta, expAvgFactor,
                dRunMean, dRunVar, 1e-5,
                dSaveMean, dSaveInvVar);
        }, warmup, iters);

        char tag[64];
        snprintf(tag, sizeof(tag), "N%d C%d %dx%d", c.B, c.C, c.H, c.W);
        double bw_cu  = bytes / (cu_ms  * 1e6);
        double bw_ref = bytes / (ref_ms * 1e6);

        results.push_back({
            std::string("BatchNorm ") + tag, tag,
            cu_ms, ref_ms, bw_cu, bw_ref
        });

        CHECK_CUDNN(cudnnDestroyTensorDescriptor(bnDesc));
        CHECK_CUDNN(cudnnDestroyTensorDescriptor(xDesc));
        CHECK_CUDA(cudaFree(dX));
        CHECK_CUDA(cudaFree(dY_cu));
        CHECK_CUDA(cudaFree(dY_ref));
        CHECK_CUDA(cudaFree(dGamma));
        CHECK_CUDA(cudaFree(dBeta));
        CHECK_CUDA(cudaFree(dRunMean));
        CHECK_CUDA(cudaFree(dRunVar));
        CHECK_CUDA(cudaFree(dSaveMean));
        CHECK_CUDA(cudaFree(dSaveInvVar));
    }

    return results;
}

// ---------------------------------------------------------------------------
// 7. Softmax — cuDeep vs cuDNN
// ---------------------------------------------------------------------------

static std::vector<BenchResult> bench_softmax() {
    std::vector<BenchResult> results;
    struct Cfg { int B, D; };
    Cfg cfgs[] = {
        {64,   128},
        {128,  512},
        {128,  4096},
        {32,   32768},
        {256,  1024},
        {1024, 1024},
    };

    for (auto& c : cfgs) {
        size_t sX = (size_t)c.B * c.D;
        float *dX, *dY_cu, *dY_ref;
        CHECK_CUDA(cudaMalloc(&dX,     sX * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dY_cu,  sX * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dY_ref, sX * sizeof(float)));
        fill_random(dX, sX);

        int warmup = 30, iters = 150;
        double bytes = (double)sX * sizeof(float) * 2;

        float cu_ms = timed_run([&]{
            cudeep::kernels::launch_softmax_kernel<float>(dX, dY_cu, c.B, c.D, 0);
        }, warmup, iters);

        // cuDNN: treat as [B, D, 1, 1] and softmax over dim C
        cudnnTensorDescriptor_t tDesc;
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&tDesc));
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(tDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                                c.B, c.D, 1, 1));
        float one = 1.0f, zero = 0.0f;
        float ref_ms = timed_run([&]{
            cudnnSoftmaxForward(g_cudnn, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
                                &one, tDesc, dX, &zero, tDesc, dY_ref);
        }, warmup, iters);

        char tag[64];
        snprintf(tag, sizeof(tag), "%dx%d", c.B, c.D);
        double bw_cu  = bytes / (cu_ms  * 1e6);
        double bw_ref = bytes / (ref_ms * 1e6);

        results.push_back({
            std::string("Softmax ") + tag, tag,
            cu_ms, ref_ms, bw_cu, bw_ref
        });

        CHECK_CUDNN(cudnnDestroyTensorDescriptor(tDesc));
        CHECK_CUDA(cudaFree(dX));
        CHECK_CUDA(cudaFree(dY_cu));
        CHECK_CUDA(cudaFree(dY_ref));
    }

    return results;
}

// ---------------------------------------------------------------------------
// 8. Reductions — cuDeep vs cuBLAS (asum/nrm2 as proxy) + manual
// ---------------------------------------------------------------------------

static std::vector<BenchResult> bench_reductions() {
    std::vector<BenchResult> results;
    int sizes[] = {1 << 17, 1 << 20, 1 << 23};  // 128K, 1M, 8M

    for (int n : sizes) {
        float *dX, *dOut;
        CHECK_CUDA(cudaMalloc(&dX,   n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dOut, sizeof(float)));
        fill_random(dX, n);

        int warmup = 30, iters = 150;
        double bytes = (double)n * sizeof(float);

        // cuDeep sum
        float cu_ms = timed_run([&]{
            cudeep::kernels::launch_sum_kernel<float>(dX, dOut, n, 0);
        }, warmup, iters);

        // cuBLAS asum (|x_i| sum — not identical but same memory pattern)
        float result_asum;
        float ref_ms = timed_run([&]{
            cublasSasum(g_cublas, n, dX, 1, &result_asum);
        }, warmup, iters);

        char tag[32];
        if (n >= (1 << 20)) snprintf(tag, sizeof(tag), "Sum %dM", n >> 20);
        else snprintf(tag, sizeof(tag), "Sum %dK", n >> 10);

        results.push_back({
            tag, tag, cu_ms, ref_ms,
            bytes / (cu_ms * 1e6), bytes / (ref_ms * 1e6)
        });

        // cuDeep max
        cu_ms = timed_run([&]{
            cudeep::kernels::launch_max_kernel<float>(dX, dOut, n, 0);
        }, warmup, iters);

        // cuBLAS isamax (index of max |x_i| — same memory access pattern)
        int result_idx;
        ref_ms = timed_run([&]{
            cublasIsamax(g_cublas, n, dX, 1, &result_idx);
        }, warmup, iters);

        if (n >= (1 << 20)) snprintf(tag, sizeof(tag), "Max %dM", n >> 20);
        else snprintf(tag, sizeof(tag), "Max %dK", n >> 10);

        results.push_back({
            tag, tag, cu_ms, ref_ms,
            bytes / (cu_ms * 1e6), bytes / (ref_ms * 1e6)
        });

        CHECK_CUDA(cudaFree(dX));
        CHECK_CUDA(cudaFree(dOut));
    }

    return results;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    int dev = 0;
    char dev_name[256] = {0};
    int sms = 0, clock_khz = 0, mem_clock_khz = 0, mem_bus_width = 0;
    int cc_major = 0, cc_minor = 0;
    size_t total_mem = 0;

    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, dev);
    cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, dev);
    cudaDeviceGetAttribute(&mem_bus_width, cudaDevAttrGlobalMemoryBusWidth, dev);
    cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, dev);
    {
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
        snprintf(dev_name, sizeof(dev_name), "%s", prop.name);
        total_mem = prop.totalGlobalMem;
    }

    float peak_fp32_gflops = (float)sms * 128 * 2 * clock_khz / 1e6f;
    float peak_bw_gbs = 2.0f * mem_clock_khz * 1e3f *
                        (mem_bus_width / 8.0f) / 1e9f;

    printf("\n");
    printf("%s\n", std::string(104, '=').c_str());
    printf("  %scuDeep vs cuBLAS + cuDNN  —  HEAD-TO-HEAD BENCHMARK%s\n", C_BOLD, C_RESET);
    printf("%s\n", std::string(104, '=').c_str());
    printf("  Device   : %s%s%s  (SM %d.%d, %d SMs @ %.0f MHz)\n",
           C_CYAN, dev_name, C_RESET, cc_major, cc_minor,
           sms, clock_khz / 1e3f);
    printf("  VRAM     : %zu MB   |   Peak FP32: %.0f GFLOPS   |   Peak BW: %.0f GB/s\n",
           total_mem >> 20, peak_fp32_gflops, peak_bw_gbs);
    printf("  cuBLAS   : cublasSgemm / cublasGemmEx\n");
    printf("  cuDNN    : %d.%d.%d\n", CUDNN_MAJOR, CUDNN_MINOR, CUDNN_PATCHLEVEL);
    printf("  Format   : %s>>> FASTER%s = cuDeep wins,  %s<<< SLOWER%s = ref wins,  %s=== TIED%s = within 3%%\n",
           C_GREEN, C_RESET, C_RED, C_RESET, C_YELLOW, C_RESET);

    CHECK_CUBLAS(cublasCreate(&g_cublas));
    CHECK_CUDNN(cudnnCreate(&g_cudnn));

    struct Section {
        const char* title;
        const char* ref_lib;
        std::function<std::vector<BenchResult>()> fn;
    };

    Section sections[] = {
        {"SGEMM (FP32 CUDA Core, no TC)", "cuBLAS", bench_gemm},
        {"GEMM (TF32 Tensor Core)",   "cuBLAS TF32", bench_gemm_tf32},
        {"Conv2d Forward",            "cuDNN",   bench_conv2d},
        {"Activations",               "cuDNN",   bench_activations},
        {"Pooling",                   "cuDNN",   bench_pooling},
        {"BatchNorm",                 "cuDNN",   bench_batchnorm},
        {"Softmax",                   "cuDNN",   bench_softmax},
        {"Reductions",                "cuBLAS",  bench_reductions},
    };

    int total = 0, total_wins = 0, total_losses = 0, total_ties = 0;
    for (auto& s : sections) {
        try {
            auto results = s.fn();
            print_section(s.title, s.ref_lib, results);
            for (auto& r : results) {
                if (r.is_win())  total_wins++;
                else if (r.is_loss()) total_losses++;
                else total_ties++;
            }
            total += (int)results.size();
        } catch (std::exception& e) {
            printf("\n  [ERROR] %s: %s\n", s.title, e.what());
        }
    }

    printf("\n%s\n", std::string(104, '=').c_str());
    printf("  %sOVERALL RESULTS  —  %d benchmarks across %zu categories%s\n",
           C_BOLD, total, sizeof(sections) / sizeof(sections[0]), C_RESET);
    printf("%s\n", std::string(104, '-').c_str());
    printf("\n");
    printf("       %s%d WINS%s      cuDeep faster than NVIDIA reference\n",
           C_GREEN, total_wins, C_RESET);
    printf("       %s%d LOSSES%s    NVIDIA reference faster\n",
           C_RED, total_losses, C_RESET);
    printf("       %s%d TIES%s      Within 3%% of each other\n",
           C_YELLOW, total_ties, C_RESET);
    printf("\n");

    double win_pct = total > 0 ? 100.0 * total_wins / total : 0;
    if (win_pct >= 60)
        printf("       %sWin rate: %.0f%%%s\n", C_GREEN, win_pct, C_RESET);
    else if (win_pct >= 40)
        printf("       %sWin rate: %.0f%%%s\n", C_YELLOW, win_pct, C_RESET);
    else
        printf("       %sWin rate: %.0f%%%s\n", C_RED, win_pct, C_RESET);

    printf("\n%s\n\n", std::string(104, '=').c_str());

    CHECK_CUDNN(cudnnDestroy(g_cudnn));
    CHECK_CUBLAS(cublasDestroy(g_cublas));
    return 0;
}
