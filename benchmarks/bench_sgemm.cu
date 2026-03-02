/*
 * bench_sgemm.cu — Honest head-to-head: cuDeep vs cuBLAS
 *
 * Two separate sections with matching precision modes:
 *   1. FP32 CUDA Core   — cuDeep sgemm_fast vs cublasSgemm (both true FP32)
 *   2. TF32 Tensor Core — cuDeep sgemm_tc   vs cuBLAS TF32  (both TF32)
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>

#include "cudeep/kernels/matmul.cuh"
#include "cudeep/error.cuh"

#define C_RESET   "\033[0m"
#define C_GREEN   "\033[1;32m"
#define C_RED     "\033[1;31m"
#define C_YELLOW  "\033[1;33m"
#define C_CYAN    "\033[1;36m"
#define C_BOLD    "\033[1m"

static void fill_random(float* d, size_t n) {
    float* h = new float[n];
    for (size_t i = 0; i < n; ++i) h[i] = (float)rand() / RAND_MAX - 0.5f;
    cudaMemcpy(d, h, n * sizeof(float), cudaMemcpyHostToDevice);
    delete[] h;
}

using LaunchFn = void(*)(const float*, const float*, float*, int, int, int, cudaStream_t);

static float bench_cudeep(LaunchFn fn, int M, int N, int K, int warmup, int iters) {
    float *A, *B, *C;
    cudaMalloc(&A, (size_t)M * K * sizeof(float));
    cudaMalloc(&B, (size_t)K * N * sizeof(float));
    cudaMalloc(&C, (size_t)M * N * sizeof(float));
    fill_random(A, (size_t)M * K);
    fill_random(B, (size_t)K * N);

    for (int i = 0; i < warmup; ++i) fn(A, B, C, M, N, K, 0);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) fn(A, B, C, M, N, K, 0);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(A); cudaFree(B); cudaFree(C);
    return ms / iters;
}

static float bench_cublas(cublasHandle_t handle, int M, int N, int K,
                          int warmup, int iters, bool tf32) {
    float *A, *B, *C;
    cudaMalloc(&A, (size_t)M * K * sizeof(float));
    cudaMalloc(&B, (size_t)K * N * sizeof(float));
    cudaMalloc(&C, (size_t)M * N * sizeof(float));
    fill_random(A, (size_t)M * K);
    fill_random(B, (size_t)K * N);

    float alpha = 1.0f, beta = 0.0f;

    auto run = [&]() {
        if (tf32) {
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                         N, M, K, &alpha,
                         B, CUDA_R_32F, N,
                         A, CUDA_R_32F, K,
                         &beta, C, CUDA_R_32F, N,
                         CUBLAS_COMPUTE_32F_FAST_TF32,
                         CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        } else {
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        N, M, K, &alpha, B, N, A, K, &beta, C, N);
        }
    };

    for (int i = 0; i < warmup; ++i) run();
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) run();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(A); cudaFree(B); cudaFree(C);
    return ms / iters;
}

static void print_row(int s, float cu_ms, float cb_ms, double flops) {
    double cu_gf = flops / (cu_ms * 1e6);
    double cb_gf = flops / (cb_ms * 1e6);
    double speedup = cb_ms / cu_ms;

    char verdict[64];
    if (cu_ms < cb_ms * 0.97)
        snprintf(verdict, sizeof(verdict), "%s>>> %.2fx FASTER%s", C_GREEN, speedup, C_RESET);
    else if (cu_ms > cb_ms * 1.03)
        snprintf(verdict, sizeof(verdict), "%s<<< %.2fx SLOWER%s", C_RED, 1.0/speedup, C_RESET);
    else
        snprintf(verdict, sizeof(verdict), "%s=== TIED%s", C_YELLOW, C_RESET);

    printf("  %6d  %9.3f  %9.1f  %9.3f  %9.1f  %s\n",
           s, cu_ms, cu_gf, cb_ms, cb_gf, verdict);
}

int main() {
    int dev = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    int sms = 0, clock_khz = 0, smem_per_sm = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, dev);
    cudaDeviceGetAttribute(&smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev);

    float peak_fp32_gflops = (float)sms * 128 * 2 * clock_khz / 1e6f;

    printf("\n%s\n", "========================================================================================================");
    printf("  %scuDeep SGEMM — Honest Benchmark vs cuBLAS%s\n", C_BOLD, C_RESET);
    printf("%s\n", "========================================================================================================");
    printf("  Device : %s%s%s  (%d SMs @ %.0f MHz)\n",
           C_CYAN, prop.name, C_RESET, sms, clock_khz / 1e3f);
    printf("  Peak FP32 : %.0f GFLOPS\n", peak_fp32_gflops);
    printf("  Shared/SM : %d KB\n", smem_per_sm / 1024);
    printf("  Format : %s>>> FASTER%s = cuDeep wins,  %s<<< SLOWER%s = cuBLAS wins,  %s=== TIED%s = within 3%%\n",
           C_GREEN, C_RESET, C_RED, C_RESET, C_YELLOW, C_RESET);

    cublasHandle_t handle;
    cublasCreate(&handle);

    int sizes[] = {128, 256, 512, 1024, 2048, 4096};

    // ---- Section 1: FP32 CUDA Core vs cuBLAS FP32 ----
    printf("\n%s\n", "--------------------------------------------------------------------------------------------------------");
    printf("  %sSECTION 1: FP32 CUDA Core%s  —  cuDeep (no TC) vs cuBLAS cublasSgemm (no TC)\n", C_BOLD, C_RESET);
    printf("  Both use true FP32 precision (23-bit mantissa). Apples-to-apples.\n");
    printf("%s\n", "--------------------------------------------------------------------------------------------------------");
    printf("  %6s  %9s  %9s  %9s  %9s  %-18s\n",
           "Size", "cuDeep ms", "cuDeep GF", "cuBLAS ms", "cuBLAS GF", "VERDICT");
    printf("  %s\n", std::string(90, '-').c_str());

    int fp32_wins = 0, fp32_losses = 0, fp32_ties = 0;
    for (int s : sizes) {
        double flops = 2.0 * s * s * s;
        int warmup = (s <= 512) ? 50 : 20;
        int iters  = (s <= 512) ? 200 : 100;

        float cu_ms = bench_cudeep(cudeep::kernels::launch_matmul_kernel_fp32<float>,
                                   s, s, s, warmup, iters);
        float cb_ms = bench_cublas(handle, s, s, s, warmup, iters, false);
        print_row(s, cu_ms, cb_ms, flops);

        if (cu_ms < cb_ms * 0.97) fp32_wins++;
        else if (cu_ms > cb_ms * 1.03) fp32_losses++;
        else fp32_ties++;
    }
    printf("  %s\n", std::string(90, '-').c_str());
    printf("  Scorecard:  %s%d WIN%s  /  %s%d LOSS%s  /  %s%d TIE%s\n",
           C_GREEN, fp32_wins, C_RESET, C_RED, fp32_losses, C_RESET,
           C_YELLOW, fp32_ties, C_RESET);

    // ---- Section 2: TF32 Tensor Core vs cuBLAS TF32 ----
    int has_tc = 0;
    cudaDeviceGetAttribute(&has_tc, cudaDevAttrComputeCapabilityMajor, dev);

    if (has_tc >= 8) {
        cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

        printf("\n%s\n", "--------------------------------------------------------------------------------------------------------");
        printf("  %sSECTION 2: TF32 Tensor Core%s  —  cuDeep TC vs cuBLAS TF32 (cublasGemmEx)\n", C_BOLD, C_RESET);
        printf("  Both use TF32 precision (~10-bit mantissa). Apples-to-apples.\n");
        printf("%s\n", "--------------------------------------------------------------------------------------------------------");
        printf("  %6s  %9s  %9s  %9s  %9s  %-18s\n",
               "Size", "cuDeep ms", "cuDeep GF", "cuBLAS ms", "cuBLAS GF", "VERDICT");
        printf("  %s\n", std::string(90, '-').c_str());

        int tf32_wins = 0, tf32_losses = 0, tf32_ties = 0;
        for (int s : sizes) {
            double flops = 2.0 * s * s * s;
            int warmup = (s <= 512) ? 50 : 20;
            int iters  = (s <= 512) ? 200 : 100;

            float cu_ms = bench_cudeep(cudeep::kernels::launch_matmul_kernel<float>,
                                       s, s, s, warmup, iters);
            float cb_ms = bench_cublas(handle, s, s, s, warmup, iters, true);
            print_row(s, cu_ms, cb_ms, flops);

            if (cu_ms < cb_ms * 0.97) tf32_wins++;
            else if (cu_ms > cb_ms * 1.03) tf32_losses++;
            else tf32_ties++;
        }
        printf("  %s\n", std::string(90, '-').c_str());
        printf("  Scorecard:  %s%d WIN%s  /  %s%d LOSS%s  /  %s%d TIE%s\n",
               C_GREEN, tf32_wins, C_RESET, C_RED, tf32_losses, C_RESET,
               C_YELLOW, tf32_ties, C_RESET);

        cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
    } else {
        printf("\n  (Tensor Cores not available on this device — SM %d.x, need SM 8.0+)\n", has_tc);
    }

    cublasDestroy(handle);
    printf("\n%s\n\n", "========================================================================================================");
    return 0;
}
