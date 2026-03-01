#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include "cudeep/ptx_intrinsics.cuh"

// Minimal test: one warp, one mma.sync.m16n8k8.row.col.f32.tf32.tf32.f32
// A is 16×8, B is 8×8, C = A * B should be 16×8

__global__ void test_mma_kernel(const float* A, const float* B, float* C) {
    int lane = threadIdx.x;
    int gid = lane / 4;
    int tg  = lane % 4;

    // TF32 A fragment (interleaved rows): a0=A[gid,tg*2] a1=A[gid+8,tg*2] a2=A[gid,tg*2+1] a3=A[gid+8,tg*2+1]
    uint32_t a0 = __float_as_uint(A[gid * 8       + tg * 2]);
    uint32_t a1 = __float_as_uint(A[(gid + 8) * 8 + tg * 2]);
    uint32_t a2 = __float_as_uint(A[gid * 8       + tg * 2 + 1]);
    uint32_t a3 = __float_as_uint(A[(gid + 8) * 8 + tg * 2 + 1]);

    // Load B fragment: B[k][n], col-major descriptor
    uint32_t b0 = __float_as_uint(B[tg * 2 * 8       + gid]);
    uint32_t b1 = __float_as_uint(B[(tg * 2 + 1) * 8 + gid]);

    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;

    cudeep::ptx::mma_m16n8k8_tf32(
        d0, d1, d2, d3,
        a0, a1, a2, a3,
        b0, b1,
        d0, d1, d2, d3);

    // D fragment (standard): d0=D[gid,tg*2] d1=D[gid,tg*2+1] d2=D[gid+8,tg*2] d3=D[gid+8,tg*2+1]
    C[gid * 8       + tg * 2]     = d0;
    C[gid * 8       + tg * 2 + 1] = d1;
    C[(gid + 8) * 8 + tg * 2]     = d2;
    C[(gid + 8) * 8 + tg * 2 + 1] = d3;
}

int main() {
    float hA[16 * 8], hB[8 * 8], hC_gpu[16 * 8], hC_ref[16 * 8];

    // Simple test: A = incrementing, B = identity-ish
    for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 8; ++j)
            hA[i * 8 + j] = (float)(i * 8 + j + 1);

    // B = identity (8×8)
    for (int i = 0; i < 64; ++i) hB[i] = 0;
    for (int i = 0; i < 8; ++i) hB[i * 8 + i] = 1.0f;

    // CPU reference: C = A * B = A (since B = I)
    for (int m = 0; m < 16; ++m)
        for (int n = 0; n < 8; ++n) {
            float sum = 0;
            for (int k = 0; k < 8; ++k)
                sum += hA[m * 8 + k] * hB[k * 8 + n];
            hC_ref[m * 8 + n] = sum;
        }

    float *dA, *dB, *dC;
    cudaMalloc(&dA, sizeof(hA));
    cudaMalloc(&dB, sizeof(hB));
    cudaMalloc(&dC, sizeof(hC_gpu));
    cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, sizeof(hC_gpu));

    test_mma_kernel<<<1, 32>>>(dA, dB, dC);
    cudaDeviceSynchronize();
    cudaMemcpy(hC_gpu, dC, sizeof(hC_gpu), cudaMemcpyDeviceToHost);

    printf("=== MMA micro-test: C = A * I (expect C == A) ===\n");
    int errs = 0;
    for (int m = 0; m < 16; ++m)
        for (int n = 0; n < 8; ++n) {
            float ref = hC_ref[m * 8 + n];
            float gpu = hC_gpu[m * 8 + n];
            float diff = fabsf(ref - gpu);
            if (diff > 0.5f) {
                if (errs < 20)
                    printf("  [%2d][%d] ref=%.1f gpu=%.1f diff=%.3f\n",
                           m, n, ref, gpu, diff);
                errs++;
            }
        }
    printf("Errors: %d / 128\n\n", errs);

    // Test 2: asymmetric random-ish data (unique values, no accidental symmetry)
    for (int i = 0; i < 128; ++i) hA[i] = (float)((i * 3 + 7) % 11) - 5.0f;
    for (int i = 0; i < 64; ++i)  hB[i] = (float)((i * 7 + 3) % 13) - 6.0f;
    for (int m = 0; m < 16; ++m)
        for (int n = 0; n < 8; ++n) {
            float sum = 0;
            for (int k = 0; k < 8; ++k)
                sum += hA[m * 8 + k] * hB[k * 8 + n];
            hC_ref[m * 8 + n] = sum;
        }
    cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, sizeof(hC_gpu));
    test_mma_kernel<<<1, 32>>>(dA, dB, dC);
    cudaDeviceSynchronize();
    cudaMemcpy(hC_gpu, dC, sizeof(hC_gpu), cudaMemcpyDeviceToHost);

    printf("=== MMA micro-test: random data ===\n");
    errs = 0;
    float max_err = 0;
    for (int m = 0; m < 16; ++m)
        for (int n = 0; n < 8; ++n) {
            float diff = fabsf(hC_ref[m*8+n] - hC_gpu[m*8+n]);
            if (diff > max_err) max_err = diff;
            if (diff > 0.5f) {
                if (errs < 10)
                    printf("  [%2d][%d] ref=%8.2f gpu=%8.2f\n",
                           m, n, hC_ref[m*8+n], hC_gpu[m*8+n]);
                errs++;
            }
        }
    printf("Max abs error: %.4f  Errors: %d / 128\n", max_err, errs);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
