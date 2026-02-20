#include "cudeep/tensor.cuh"
#include "cudeep/error.cuh"

#include <cassert>
#include <cstdio>
#include <vector>
#include <cmath>

using namespace cudeep;

static constexpr float EPS = 1e-4f;

// ---- Creation / metadata ----

void test_tensor_create() {
    Tensor t({2, 3, 4});
    assert(t.ndim() == 3);
    assert(t.numel() == 24);
    assert(t.nbytes() == 24 * sizeof(float));
    assert(t.is_contiguous());
    printf("[PASS] test_tensor_create\n");
}

void test_tensor_zeros() {
    Tensor t = Tensor::zeros({4, 4});
    std::vector<float> host(16, -1.0f);
    t.to_host(host.data());
    for (int i = 0; i < 16; ++i) {
        assert(host[i] == 0.0f);
    }
    printf("[PASS] test_tensor_zeros\n");
}

void test_tensor_ones() {
    Tensor t = Tensor::ones({3, 3});
    std::vector<float> host(9);
    t.to_host(host.data());
    for (int i = 0; i < 9; ++i) {
        assert(fabs(host[i] - 1.0f) < EPS);
    }
    printf("[PASS] test_tensor_ones\n");
}

void test_tensor_randn() {
    Tensor t = Tensor::randn({1000});
    std::vector<float> host(1000);
    t.to_host(host.data());

    float sum = 0;
    for (float v : host) sum += v;
    float mean = sum / 1000.0f;
    assert(fabs(mean) < 0.2f);

    bool all_same = true;
    for (int i = 1; i < 1000; ++i) {
        if (host[i] != host[0]) { all_same = false; break; }
    }
    assert(!all_same);
    printf("[PASS] test_tensor_randn\n");
}

void test_tensor_randn_odd() {
    Tensor t = Tensor::randn({7});
    std::vector<float> host(7);
    t.to_host(host.data());
    for (float v : host) {
        assert(std::isfinite(v));
    }
    printf("[PASS] test_tensor_randn_odd\n");
}

void test_tensor_from_host() {
    std::vector<float> data = {1, 2, 3, 4, 5, 6};
    Tensor t = Tensor::from_host(data.data(), {2, 3});

    std::vector<float> out(6);
    t.to_host(out.data());
    for (int i = 0; i < 6; ++i) {
        assert(out[i] == data[i]);
    }
    printf("[PASS] test_tensor_from_host\n");
}

// ---- Copy / move ----

void test_tensor_copy() {
    std::vector<float> data = {1, 2, 3, 4};
    Tensor a = Tensor::from_host(data.data(), {2, 2});
    Tensor b = a;

    std::vector<float> out(4);
    b.to_host(out.data());
    for (int i = 0; i < 4; ++i) {
        assert(out[i] == data[i]);
    }
    printf("[PASS] test_tensor_copy\n");
}

void test_tensor_move() {
    Tensor a = Tensor::ones({3, 3});
    Tensor b = std::move(a);

    assert(b.numel() == 9);
    std::vector<float> out(9);
    b.to_host(out.data());
    for (int i = 0; i < 9; ++i) {
        assert(fabs(out[i] - 1.0f) < EPS);
    }
    printf("[PASS] test_tensor_move\n");
}

// ---- Reshape ----

void test_tensor_reshape() {
    Tensor t({2, 3, 4});
    Tensor r = t.reshape({6, 4});
    assert(r.shape()[0] == 6);
    assert(r.shape()[1] == 4);
    assert(r.numel() == 24);
    printf("[PASS] test_tensor_reshape\n");
}

// ---- Elementwise operators ----

void test_tensor_add() {
    std::vector<float> a = {1, 2, 3, 4};
    std::vector<float> b = {10, 20, 30, 40};
    Tensor ta = Tensor::from_host(a.data(), {4});
    Tensor tb = Tensor::from_host(b.data(), {4});
    Tensor tc = ta + tb;

    std::vector<float> c(4);
    tc.to_host(c.data());
    assert(fabs(c[0] - 11.0f) < EPS);
    assert(fabs(c[1] - 22.0f) < EPS);
    assert(fabs(c[2] - 33.0f) < EPS);
    assert(fabs(c[3] - 44.0f) < EPS);
    printf("[PASS] test_tensor_add\n");
}

void test_tensor_sub() {
    std::vector<float> a = {10, 20, 30};
    std::vector<float> b = {1, 2, 3};
    Tensor ta = Tensor::from_host(a.data(), {3});
    Tensor tb = Tensor::from_host(b.data(), {3});
    Tensor tc = ta - tb;

    std::vector<float> c(3);
    tc.to_host(c.data());
    assert(fabs(c[0] - 9.0f) < EPS);
    assert(fabs(c[1] - 18.0f) < EPS);
    assert(fabs(c[2] - 27.0f) < EPS);
    printf("[PASS] test_tensor_sub\n");
}

void test_tensor_mul() {
    std::vector<float> a = {2, 3, 4};
    std::vector<float> b = {5, 6, 7};
    Tensor ta = Tensor::from_host(a.data(), {3});
    Tensor tb = Tensor::from_host(b.data(), {3});
    Tensor tc = ta * tb;

    std::vector<float> c(3);
    tc.to_host(c.data());
    assert(fabs(c[0] - 10.0f) < EPS);
    assert(fabs(c[1] - 18.0f) < EPS);
    assert(fabs(c[2] - 28.0f) < EPS);
    printf("[PASS] test_tensor_mul\n");
}

void test_tensor_chained_ops() {
    std::vector<float> a = {1, 2};
    std::vector<float> b = {3, 4};
    std::vector<float> c = {5, 6};
    Tensor ta = Tensor::from_host(a.data(), {2});
    Tensor tb = Tensor::from_host(b.data(), {2});
    Tensor tc = Tensor::from_host(c.data(), {2});

    Tensor result = (ta + tb) * tc;
    std::vector<float> out(2);
    result.to_host(out.data());
    assert(fabs(out[0] - (1+3)*5) < EPS);
    assert(fabs(out[1] - (2+4)*6) < EPS);
    printf("[PASS] test_tensor_chained_ops\n");
}

// ---- Matmul ----

void test_tensor_matmul() {
    std::vector<float> A = {1, 2, 3, 4};
    std::vector<float> B = {5, 6, 7, 8};
    Tensor tA = Tensor::from_host(A.data(), {2, 2});
    Tensor tB = Tensor::from_host(B.data(), {2, 2});
    Tensor tC = tA.matmul(tB);

    assert(tC.shape()[0] == 2);
    assert(tC.shape()[1] == 2);

    std::vector<float> C(4);
    tC.to_host(C.data());
    assert(fabs(C[0] - 19.0f) < EPS);
    assert(fabs(C[1] - 22.0f) < EPS);
    assert(fabs(C[2] - 43.0f) < EPS);
    assert(fabs(C[3] - 50.0f) < EPS);
    printf("[PASS] test_tensor_matmul\n");
}

void test_tensor_matmul_nonsquare() {
    // [2,3] x [3,4] = [2,4]
    std::vector<float> A = {1, 2, 3, 4, 5, 6};
    std::vector<float> B = {1, 0, 0, 0,
                            0, 1, 0, 0,
                            0, 0, 1, 0};
    Tensor tA = Tensor::from_host(A.data(), {2, 3});
    Tensor tB = Tensor::from_host(B.data(), {3, 4});
    Tensor tC = tA.matmul(tB);

    assert(tC.shape()[0] == 2);
    assert(tC.shape()[1] == 4);

    std::vector<float> C(8);
    tC.to_host(C.data());
    // First row of A times columns of B (partial identity)
    assert(fabs(C[0] - 1.0f) < EPS);
    assert(fabs(C[1] - 2.0f) < EPS);
    assert(fabs(C[2] - 3.0f) < EPS);
    assert(fabs(C[3] - 0.0f) < EPS);
    printf("[PASS] test_tensor_matmul_nonsquare\n");
}

void test_tensor_matmul_identity() {
    const int N = 8;
    std::vector<float> A_data(N * N);
    for (int i = 0; i < N * N; ++i) A_data[i] = static_cast<float>(i) * 0.1f;

    std::vector<float> I_data(N * N, 0.0f);
    for (int i = 0; i < N; ++i) I_data[i * N + i] = 1.0f;

    Tensor tA = Tensor::from_host(A_data.data(), {N, N});
    Tensor tI = Tensor::from_host(I_data.data(), {N, N});
    Tensor tC = tA.matmul(tI);

    std::vector<float> C(N * N);
    tC.to_host(C.data());
    for (int i = 0; i < N * N; ++i) {
        assert(fabs(C[i] - A_data[i]) < 0.01f);
    }
    printf("[PASS] test_tensor_matmul_identity\n");
}

// ---- Transpose ----

void test_tensor_transpose_shape() {
    Tensor t({2, 3});
    Tensor tt = t.transpose(0, 1);
    assert(tt.shape()[0] == 3);
    assert(tt.shape()[1] == 2);
    printf("[PASS] test_tensor_transpose_shape\n");
}

void test_tensor_transpose_not_contiguous() {
    Tensor t({2, 3});
    Tensor tt = t.transpose(0, 1);
    assert(!tt.is_contiguous());
    printf("[PASS] test_tensor_transpose_not_contiguous\n");
}

void test_tensor_transpose_3d() {
    Tensor t({2, 3, 4});
    Tensor tt = t.transpose(0, 2);
    assert(tt.shape()[0] == 4);
    assert(tt.shape()[1] == 3);
    assert(tt.shape()[2] == 2);
    printf("[PASS] test_tensor_transpose_3d\n");
}

// ---- Contiguous ----

void test_tensor_contiguous_noop() {
    std::vector<float> data = {1, 2, 3, 4};
    Tensor t = Tensor::from_host(data.data(), {2, 2});
    assert(t.is_contiguous());
    Tensor tc = t.contiguous();
    assert(tc.is_contiguous());

    std::vector<float> out(4);
    tc.to_host(out.data());
    for (int i = 0; i < 4; ++i) {
        assert(out[i] == data[i]);
    }
    printf("[PASS] test_tensor_contiguous_noop\n");
}

void test_tensor_contiguous_after_transpose() {
    // [2,3] data:
    // 1 2 3
    // 4 5 6
    std::vector<float> data = {1, 2, 3, 4, 5, 6};
    Tensor t = Tensor::from_host(data.data(), {2, 3});
    Tensor tt = t.transpose(0, 1);
    assert(!tt.is_contiguous());
    assert(tt.shape()[0] == 3);
    assert(tt.shape()[1] == 2);

    Tensor tc = tt.contiguous();
    assert(tc.is_contiguous());
    assert(tc.shape()[0] == 3);
    assert(tc.shape()[1] == 2);

    // Transposed should be:
    // 1 4
    // 2 5
    // 3 6
    std::vector<float> out(6);
    tc.to_host(out.data());
    assert(fabs(out[0] - 1.0f) < EPS);
    assert(fabs(out[1] - 4.0f) < EPS);
    assert(fabs(out[2] - 2.0f) < EPS);
    assert(fabs(out[3] - 5.0f) < EPS);
    assert(fabs(out[4] - 3.0f) < EPS);
    assert(fabs(out[5] - 6.0f) < EPS);
    printf("[PASS] test_tensor_contiguous_after_transpose\n");
}

// ---- Fill / zero ----

void test_tensor_fill() {
    Tensor t({2, 3});
    t.fill_(42.0f);

    std::vector<float> out(6);
    t.to_host(out.data());
    for (int i = 0; i < 6; ++i) {
        assert(fabs(out[i] - 42.0f) < EPS);
    }
    printf("[PASS] test_tensor_fill\n");
}

void test_tensor_zero() {
    Tensor t = Tensor::ones({4});
    t.zero_();

    std::vector<float> out(4);
    t.to_host(out.data());
    for (int i = 0; i < 4; ++i) {
        assert(out[i] == 0.0f);
    }
    printf("[PASS] test_tensor_zero\n");
}

// ---- f64 tests ----

void test_tensor_f64_ops() {
    std::vector<double> a = {1.0, 2.0, 3.0};
    std::vector<double> b = {10.0, 20.0, 30.0};
    Tensor ta = Tensor::from_host(a.data(), {3}, DType::Float64);
    Tensor tb = Tensor::from_host(b.data(), {3}, DType::Float64);

    Tensor tc = ta + tb;
    std::vector<double> c(3);
    tc.to_host(c.data());
    assert(fabs(c[0] - 11.0) < 1e-10);
    assert(fabs(c[1] - 22.0) < 1e-10);
    assert(fabs(c[2] - 33.0) < 1e-10);
    printf("[PASS] test_tensor_f64_ops\n");
}

void test_tensor_f64_matmul() {
    std::vector<double> A = {1, 2, 3, 4};
    std::vector<double> B = {5, 6, 7, 8};
    Tensor tA = Tensor::from_host(A.data(), {2, 2}, DType::Float64);
    Tensor tB = Tensor::from_host(B.data(), {2, 2}, DType::Float64);
    Tensor tC = tA.matmul(tB);

    std::vector<double> C(4);
    tC.to_host(C.data());
    assert(fabs(C[0] - 19.0) < 1e-10);
    assert(fabs(C[1] - 22.0) < 1e-10);
    assert(fabs(C[2] - 43.0) < 1e-10);
    assert(fabs(C[3] - 50.0) < 1e-10);
    printf("[PASS] test_tensor_f64_matmul\n");
}

int main() {
    printf("=== Creation / Metadata ===\n");
    test_tensor_create();
    test_tensor_zeros();
    test_tensor_ones();
    test_tensor_randn();
    test_tensor_randn_odd();
    test_tensor_from_host();

    printf("\n=== Copy / Move ===\n");
    test_tensor_copy();
    test_tensor_move();

    printf("\n=== Reshape ===\n");
    test_tensor_reshape();

    printf("\n=== Elementwise Operators ===\n");
    test_tensor_add();
    test_tensor_sub();
    test_tensor_mul();
    test_tensor_chained_ops();

    printf("\n=== Matmul ===\n");
    test_tensor_matmul();
    test_tensor_matmul_nonsquare();
    test_tensor_matmul_identity();

    printf("\n=== Transpose ===\n");
    test_tensor_transpose_shape();
    test_tensor_transpose_not_contiguous();
    test_tensor_transpose_3d();

    printf("\n=== Contiguous ===\n");
    test_tensor_contiguous_noop();
    test_tensor_contiguous_after_transpose();

    printf("\n=== Fill / Zero ===\n");
    test_tensor_fill();
    test_tensor_zero();

    printf("\n=== Float64 ===\n");
    test_tensor_f64_ops();
    test_tensor_f64_matmul();

    printf("\nAll tensor tests passed!\n");
    return 0;
}
