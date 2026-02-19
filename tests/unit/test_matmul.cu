#include "cudeep/tensor.cuh"
#include "cudeep/memory.cuh"
#include "cudeep/kernels/matmul.cuh"
#include "cudeep/error.cuh"

#include <cassert>
#include <cstdio>
#include <vector>
#include <cmath>

using namespace cudeep;

void test_matmul_identity() {
    const int N = 4;

    std::vector<float> A_data(N * N, 0.0f);
    std::vector<float> I_data(N * N, 0.0f);
    for (int i = 0; i < N; ++i) {
        A_data[i * N + i] = static_cast<float>(i + 1);
        I_data[i * N + i] = 1.0f;
    }

    Tensor A = Tensor::from_host(A_data.data(), {N, N});
    Tensor I = Tensor::from_host(I_data.data(), {N, N});
    Tensor C = Tensor::zeros({N, N});

    kernels::launch_matmul_kernel<float>(
        static_cast<const float*>(A.data()),
        static_cast<const float*>(I.data()),
        static_cast<float*>(C.data()),
        N, N, N
    );
    CUDEEP_CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> result(N * N);
    C.to_host(result.data());

    for (int i = 0; i < N * N; ++i) {
        assert(fabs(result[i] - A_data[i]) < 1e-5f);
    }
    printf("[PASS] test_matmul_identity\n");
}

void test_matmul_2x2() {
    std::vector<float> A = {1, 2, 3, 4};
    std::vector<float> B = {5, 6, 7, 8};

    Tensor tA = Tensor::from_host(A.data(), {2, 2});
    Tensor tB = Tensor::from_host(B.data(), {2, 2});
    Tensor tC = Tensor::zeros({2, 2});

    kernels::launch_matmul_kernel<float>(
        static_cast<const float*>(tA.data()),
        static_cast<const float*>(tB.data()),
        static_cast<float*>(tC.data()),
        2, 2, 2
    );
    CUDEEP_CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> C(4);
    tC.to_host(C.data());

    assert(fabs(C[0] - 19.0f) < 1e-5f);
    assert(fabs(C[1] - 22.0f) < 1e-5f);
    assert(fabs(C[2] - 43.0f) < 1e-5f);
    assert(fabs(C[3] - 50.0f) < 1e-5f);
    printf("[PASS] test_matmul_2x2\n");
}

int main() {
    test_matmul_identity();
    test_matmul_2x2();
    printf("\nAll matmul tests passed!\n");
    return 0;
}
