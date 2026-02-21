#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>

#include "cudeep/kernels/matmul.cuh"
#include "cudeep/error.cuh"

using cudeep::kernels::launch_matmul_kernel;

static void fill_random(float* d, int n) {
    float* h = new float[n];
    for (int i = 0; i < n; ++i) h[i] = (float)rand() / RAND_MAX - 0.5f;
    cudaMemcpy(d, h, n * sizeof(float), cudaMemcpyHostToDevice);
    delete[] h;
}

static float bench_cudeep(int M, int N, int K, int warmup, int iters) {
    float *A, *B, *C;
    cudaMalloc(&A, (size_t)M * K * sizeof(float));
    cudaMalloc(&B, (size_t)K * N * sizeof(float));
    cudaMalloc(&C, (size_t)M * N * sizeof(float));
    fill_random(A, M * K);
    fill_random(B, K * N);

    for (int i = 0; i < warmup; ++i)
        launch_matmul_kernel<float>(A, B, C, M, N, K, 0);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i)
        launch_matmul_kernel<float>(A, B, C, M, N, K, 0);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= iters;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(A); cudaFree(B); cudaFree(C);
    return ms;
}

static float bench_cublas(int M, int N, int K, int warmup, int iters) {
    float *A, *B, *C;
    cudaMalloc(&A, (size_t)M * K * sizeof(float));
    cudaMalloc(&B, (size_t)K * N * sizeof(float));
    cudaMalloc(&C, (size_t)M * N * sizeof(float));
    fill_random(A, M * K);
    fill_random(B, K * N);

    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;

    // cuBLAS expects column-major, but for row-major C = A*B:
    // C^T = B^T * A^T → cublasSgemm(N, N, N, K, M, ..., B, N, A, K, ..., C, N)
    for (int i = 0; i < warmup; ++i)
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                    &alpha, B, N, A, K, &beta, C, N);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i)
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                    &alpha, B, N, A, K, &beta, C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= iters;

    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(A); cudaFree(B); cudaFree(C);
    return ms;
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    float peak_tflops = (float)prop.multiProcessorCount * 128 * 2 *
                        prop.clockRate / 1e6f;
    printf("Device: %s (%d SMs, %.0f MHz)\n", prop.name,
           prop.multiProcessorCount, prop.clockRate / 1e3f);
    printf("Theoretical FP32 peak: %.1f GFLOPS\n", peak_tflops);
    printf("Shared mem/SM: %zu KB\n\n", prop.sharedMemPerMultiprocessor / 1024);

    printf("%8s  %10s  %10s  %10s  %10s  %8s  %8s\n",
           "Size", "cuDeep ms", "cuDeep GF", "cuBLAS ms", "cuBLAS GF",
           "Ratio", "% Peak");
    printf("---------------------------------------------------------------------\n");

    int sizes[] = {128, 256, 512, 1024, 2048, 4096};
    for (int s : sizes) {
        double flops = 2.0 * s * s * s;
        int warmup = (s <= 512) ? 50 : 20;
        int iters = (s <= 512) ? 200 : 100;

        float cu_ms = bench_cudeep(s, s, s, warmup, iters);
        float cb_ms = bench_cublas(s, s, s, warmup, iters);
        double cu_gf = flops / (cu_ms * 1e6);
        double cb_gf = flops / (cb_ms * 1e6);
        double ratio = cu_gf / cb_gf;
        double pct = cu_gf / peak_tflops * 100.0;

        printf("%8d  %10.3f  %10.1f  %10.3f  %10.1f  %7.1f%%  %7.1f%%\n",
               s, cu_ms, cu_gf, cb_ms, cb_gf, ratio * 100, pct);
    }
    return 0;
}
