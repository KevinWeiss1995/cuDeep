#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "cudeep/kernels/matmul.cuh"

using cudeep::kernels::launch_matmul_kernel;

int main() {
    // Verify TC kernel correctness against cuBLAS reference
    int sizes[] = {128, 256, 512, 1024, 2048};
    cublasHandle_t handle;
    cublasCreate(&handle);

    for (int S : sizes) {
        int n = S * S;
        float *hA = new float[S * S];
        float *hB = new float[S * S];
        float *hC_us = new float[S * S];
        float *hC_ref = new float[S * S];

        srand(42);
        for (int i = 0; i < n; ++i) {
            hA[i] = (float)rand() / RAND_MAX - 0.5f;
            hB[i] = (float)rand() / RAND_MAX - 0.5f;
        }

        float *dA, *dB, *dC_us, *dC_ref;
        cudaMalloc(&dA, n * sizeof(float));
        cudaMalloc(&dB, n * sizeof(float));
        cudaMalloc(&dC_us, n * sizeof(float));
        cudaMalloc(&dC_ref, n * sizeof(float));

        cudaMemcpy(dA, hA, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB, n * sizeof(float), cudaMemcpyHostToDevice);

        // Our kernel
        launch_matmul_kernel<float>(dA, dB, dC_us, S, S, S, 0);
        cudaDeviceSynchronize();

        // cuBLAS reference (row-major: C = A*B via C^T = B^T * A^T)
        float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, S, S, S,
                    &alpha, dB, S, dA, S, &beta, dC_ref, S);
        cudaDeviceSynchronize();

        cudaMemcpy(hC_us, dC_us, n * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hC_ref, dC_ref, n * sizeof(float), cudaMemcpyDeviceToHost);

        // TF32 has 10-bit mantissa → ~1e-3 relative error expected
        float max_rel_err = 0, max_abs_err = 0;
        int errs = 0;
        for (int i = 0; i < n; ++i) {
            float diff = fabsf(hC_us[i] - hC_ref[i]);
            float ref_abs = fabsf(hC_ref[i]);
            float rel = (ref_abs > 1e-6f) ? diff / ref_abs : diff;
            if (diff > max_abs_err) max_abs_err = diff;
            if (rel > max_rel_err) max_rel_err = rel;
            if (rel > 0.05f) errs++;
        }

        printf("[%4d×%4d]  max_abs=%.6f  max_rel=%.6f  errors(>5%%)=%d  %s\n",
               S, S, max_abs_err, max_rel_err, errs,
               (errs == 0) ? "PASS" : "FAIL");

        cudaFree(dA); cudaFree(dB); cudaFree(dC_us); cudaFree(dC_ref);
        delete[] hA; delete[] hB; delete[] hC_us; delete[] hC_ref;
    }
    cublasDestroy(handle);
    return 0;
}
