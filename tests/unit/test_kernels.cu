#include "cudeep/tensor.cuh"
#include "cudeep/memory.cuh"
#include "cudeep/error.cuh"
#include "cudeep/kernels/elementwise.cuh"
#include "cudeep/kernels/activation.cuh"
#include "cudeep/kernels/reduce.cuh"
#include "cudeep/kernels/loss.cuh"
#include "cudeep/kernels/conv.cuh"

#include <cassert>
#include <cstdio>
#include <cmath>
#include <vector>
#include <numeric>

using namespace cudeep;

static constexpr float EPS = 1e-4f;

static float* to_device(const std::vector<float>& v) {
    float* d = static_cast<float*>(device_malloc(v.size() * sizeof(float)));
    memcpy_h2d(d, v.data(), v.size() * sizeof(float));
    return d;
}

static std::vector<float> to_host(const float* d, size_t n) {
    std::vector<float> h(n);
    memcpy_d2h(h.data(), d, n * sizeof(float));
    return h;
}

// ---- Elementwise tests ----

void test_add_kernel() {
    std::vector<float> a = {1, 2, 3, 4};
    std::vector<float> b = {10, 20, 30, 40};
    float* da = to_device(a);
    float* db = to_device(b);
    float* dc = static_cast<float*>(device_malloc(4 * sizeof(float)));

    kernels::launch_add_kernel<float>(da, db, dc, 4);
    CUDEEP_CHECK_CUDA(cudaDeviceSynchronize());

    auto c = to_host(dc, 4);
    assert(fabs(c[0] - 11.0f) < EPS);
    assert(fabs(c[1] - 22.0f) < EPS);
    assert(fabs(c[2] - 33.0f) < EPS);
    assert(fabs(c[3] - 44.0f) < EPS);

    device_free(da); device_free(db); device_free(dc);
    printf("[PASS] test_add_kernel\n");
}

void test_sub_kernel() {
    std::vector<float> a = {10, 20, 30};
    std::vector<float> b = {1, 2, 3};
    float* da = to_device(a);
    float* db = to_device(b);
    float* dc = static_cast<float*>(device_malloc(3 * sizeof(float)));

    kernels::launch_sub_kernel<float>(da, db, dc, 3);
    CUDEEP_CHECK_CUDA(cudaDeviceSynchronize());

    auto c = to_host(dc, 3);
    assert(fabs(c[0] - 9.0f) < EPS);
    assert(fabs(c[1] - 18.0f) < EPS);
    assert(fabs(c[2] - 27.0f) < EPS);

    device_free(da); device_free(db); device_free(dc);
    printf("[PASS] test_sub_kernel\n");
}

void test_mul_kernel() {
    std::vector<float> a = {2, 3, 4};
    std::vector<float> b = {5, 6, 7};
    float* da = to_device(a);
    float* db = to_device(b);
    float* dc = static_cast<float*>(device_malloc(3 * sizeof(float)));

    kernels::launch_mul_kernel<float>(da, db, dc, 3);
    CUDEEP_CHECK_CUDA(cudaDeviceSynchronize());

    auto c = to_host(dc, 3);
    assert(fabs(c[0] - 10.0f) < EPS);
    assert(fabs(c[1] - 18.0f) < EPS);
    assert(fabs(c[2] - 28.0f) < EPS);

    device_free(da); device_free(db); device_free(dc);
    printf("[PASS] test_mul_kernel\n");
}

void test_fill_kernel() {
    float* d = static_cast<float*>(device_malloc(8 * sizeof(float)));
    kernels::launch_fill_kernel<float>(d, 7.5f, 8);
    CUDEEP_CHECK_CUDA(cudaDeviceSynchronize());

    auto h = to_host(d, 8);
    for (int i = 0; i < 8; ++i) {
        assert(fabs(h[i] - 7.5f) < EPS);
    }
    device_free(d);
    printf("[PASS] test_fill_kernel\n");
}

void test_scalar_mul_kernel() {
    std::vector<float> a = {2, 4, 6};
    float* da = to_device(a);
    float* dc = static_cast<float*>(device_malloc(3 * sizeof(float)));

    kernels::launch_scalar_mul_kernel<float>(da, 0.5f, dc, 3);
    CUDEEP_CHECK_CUDA(cudaDeviceSynchronize());

    auto c = to_host(dc, 3);
    assert(fabs(c[0] - 1.0f) < EPS);
    assert(fabs(c[1] - 2.0f) < EPS);
    assert(fabs(c[2] - 3.0f) < EPS);

    device_free(da); device_free(dc);
    printf("[PASS] test_scalar_mul_kernel\n");
}

// ---- Activation tests ----

void test_relu_forward() {
    std::vector<float> in = {-3, -1, 0, 1, 3};
    float* din = to_device(in);
    float* dout = static_cast<float*>(device_malloc(5 * sizeof(float)));

    kernels::launch_activation_forward_kernel<float>(
        din, dout, 5, kernels::ActivationType::ReLU);
    CUDEEP_CHECK_CUDA(cudaDeviceSynchronize());

    auto out = to_host(dout, 5);
    assert(out[0] == 0.0f);
    assert(out[1] == 0.0f);
    assert(out[2] == 0.0f);
    assert(fabs(out[3] - 1.0f) < EPS);
    assert(fabs(out[4] - 3.0f) < EPS);

    device_free(din); device_free(dout);
    printf("[PASS] test_relu_forward\n");
}

void test_sigmoid_forward() {
    std::vector<float> in = {0.0f};
    float* din = to_device(in);
    float* dout = static_cast<float*>(device_malloc(sizeof(float)));

    kernels::launch_activation_forward_kernel<float>(
        din, dout, 1, kernels::ActivationType::Sigmoid);
    CUDEEP_CHECK_CUDA(cudaDeviceSynchronize());

    auto out = to_host(dout, 1);
    assert(fabs(out[0] - 0.5f) < EPS);

    device_free(din); device_free(dout);
    printf("[PASS] test_sigmoid_forward\n");
}

void test_gelu_forward() {
    std::vector<float> in = {0.0f, 1.0f, -1.0f};
    float* din = to_device(in);
    float* dout = static_cast<float*>(device_malloc(3 * sizeof(float)));

    kernels::launch_activation_forward_kernel<float>(
        din, dout, 3, kernels::ActivationType::GELU);
    CUDEEP_CHECK_CUDA(cudaDeviceSynchronize());

    auto out = to_host(dout, 3);
    assert(fabs(out[0] - 0.0f) < EPS);  // GELU(0) = 0
    assert(out[1] > 0.8f && out[1] < 0.85f);  // GELU(1) ≈ 0.841
    assert(out[2] > -0.17f && out[2] < -0.15f);  // GELU(-1) ≈ -0.159

    device_free(din); device_free(dout);
    printf("[PASS] test_gelu_forward\n");
}

void test_relu_backward() {
    std::vector<float> in = {-2, 0.5f, 3};
    std::vector<float> grad_out = {1, 1, 1};
    float* din = to_device(in);
    float* dgo = to_device(grad_out);
    float* dgi = static_cast<float*>(device_malloc(3 * sizeof(float)));

    kernels::launch_activation_backward_kernel<float>(
        dgo, din, dgi, 3, kernels::ActivationType::ReLU);
    CUDEEP_CHECK_CUDA(cudaDeviceSynchronize());

    auto gi = to_host(dgi, 3);
    assert(gi[0] == 0.0f);
    assert(fabs(gi[1] - 1.0f) < EPS);
    assert(fabs(gi[2] - 1.0f) < EPS);

    device_free(din); device_free(dgo); device_free(dgi);
    printf("[PASS] test_relu_backward\n");
}

void test_leaky_relu_forward() {
    std::vector<float> in = {-10, 5};
    float* din = to_device(in);
    float* dout = static_cast<float*>(device_malloc(2 * sizeof(float)));

    kernels::launch_activation_forward_kernel<float>(
        din, dout, 2, kernels::ActivationType::LeakyReLU, 0.1f);
    CUDEEP_CHECK_CUDA(cudaDeviceSynchronize());

    auto out = to_host(dout, 2);
    assert(fabs(out[0] - (-1.0f)) < EPS);
    assert(fabs(out[1] - 5.0f) < EPS);

    device_free(din); device_free(dout);
    printf("[PASS] test_leaky_relu_forward\n");
}

// ---- Reduction tests ----

void test_sum_kernel() {
    std::vector<float> data = {1, 2, 3, 4, 5};
    float* din = to_device(data);
    float* dout = static_cast<float*>(device_malloc(sizeof(float)));

    kernels::launch_sum_kernel<float>(din, dout, 5);
    CUDEEP_CHECK_CUDA(cudaDeviceSynchronize());

    auto out = to_host(dout, 1);
    assert(fabs(out[0] - 15.0f) < EPS);

    device_free(din); device_free(dout);
    printf("[PASS] test_sum_kernel\n");
}

void test_sum_large() {
    const int N = 4096;
    std::vector<float> data(N, 1.0f);
    float* din = to_device(data);
    float* dout = static_cast<float*>(device_malloc(sizeof(float)));

    kernels::launch_sum_kernel<float>(din, dout, N);
    CUDEEP_CHECK_CUDA(cudaDeviceSynchronize());

    auto out = to_host(dout, 1);
    assert(fabs(out[0] - static_cast<float>(N)) < 1.0f);

    device_free(din); device_free(dout);
    printf("[PASS] test_sum_large\n");
}

// ---- Loss tests ----

void test_mse_loss_zero() {
    std::vector<float> pred = {1, 2, 3, 4};
    std::vector<float> target = {1, 2, 3, 4};
    float* dp = to_device(pred);
    float* dt = to_device(target);
    float* dl = static_cast<float*>(device_malloc(sizeof(float)));

    kernels::launch_mse_loss_kernel<float>(dp, dt, dl, 4);
    CUDEEP_CHECK_CUDA(cudaDeviceSynchronize());

    auto loss = to_host(dl, 1);
    assert(fabs(loss[0]) < EPS);

    device_free(dp); device_free(dt); device_free(dl);
    printf("[PASS] test_mse_loss_zero\n");
}

void test_mse_loss_nonzero() {
    std::vector<float> pred = {1, 2, 3};
    std::vector<float> target = {2, 3, 4};
    float* dp = to_device(pred);
    float* dt = to_device(target);
    float* dl = static_cast<float*>(device_malloc(sizeof(float)));

    kernels::launch_mse_loss_kernel<float>(dp, dt, dl, 3);
    CUDEEP_CHECK_CUDA(cudaDeviceSynchronize());

    auto loss = to_host(dl, 1);
    // MSE = mean((1)^2 + (1)^2 + (1)^2) = 3/3 = 1.0
    // But kernel does atomicAdd(shared[0]/n) per block, so for 1 block: 3/3 = 1.0
    assert(fabs(loss[0] - 1.0f) < EPS);

    device_free(dp); device_free(dt); device_free(dl);
    printf("[PASS] test_mse_loss_nonzero\n");
}

void test_softmax_kernel() {
    std::vector<float> in = {1, 2, 3};  // 1 batch, dim=3
    float* din = to_device(in);
    float* dout = static_cast<float*>(device_malloc(3 * sizeof(float)));

    kernels::launch_softmax_kernel<float>(din, dout, 1, 3);
    CUDEEP_CHECK_CUDA(cudaDeviceSynchronize());

    auto out = to_host(dout, 3);
    float sum = out[0] + out[1] + out[2];
    assert(fabs(sum - 1.0f) < EPS);
    assert(out[0] < out[1] && out[1] < out[2]);

    device_free(din); device_free(dout);
    printf("[PASS] test_softmax_kernel\n");
}

void test_softmax_batch() {
    // 2 batches, dim=2
    std::vector<float> in = {0, 0, 100, 0};
    float* din = to_device(in);
    float* dout = static_cast<float*>(device_malloc(4 * sizeof(float)));

    kernels::launch_softmax_kernel<float>(din, dout, 2, 2);
    CUDEEP_CHECK_CUDA(cudaDeviceSynchronize());

    auto out = to_host(dout, 4);
    assert(fabs(out[0] - 0.5f) < EPS);
    assert(fabs(out[1] - 0.5f) < EPS);
    assert(out[2] > 0.99f);
    assert(out[3] < 0.01f);

    device_free(din); device_free(dout);
    printf("[PASS] test_softmax_batch\n");
}

// ---- Conv2d forward test ----

void test_conv2d_forward_identity() {
    // 1 batch, 1 in_ch, 4x4 input, 1 out_ch, 1x1 kernel (weight=1), no padding, stride 1
    const int B = 1, IC = 1, OC = 1, H = 4, W = 4, KH = 1, KW = 1;
    std::vector<float> input(B * IC * H * W);
    std::iota(input.begin(), input.end(), 1.0f);
    std::vector<float> weight = {1.0f};
    std::vector<float> bias = {0.0f};

    float* din = to_device(input);
    float* dw = to_device(weight);
    float* db = to_device(bias);

    int out_h = H, out_w = W;
    float* dout = static_cast<float*>(device_malloc(B * OC * out_h * out_w * sizeof(float)));

    kernels::launch_conv2d_forward_kernel<float>(
        din, dw, db, dout, B, IC, OC, H, W, KH, KW, 1, 1, 0, 0);
    CUDEEP_CHECK_CUDA(cudaDeviceSynchronize());

    auto out = to_host(dout, B * OC * out_h * out_w);
    for (int i = 0; i < static_cast<int>(input.size()); ++i) {
        assert(fabs(out[i] - input[i]) < EPS);
    }

    device_free(din); device_free(dw); device_free(db); device_free(dout);
    printf("[PASS] test_conv2d_forward_identity\n");
}

void test_conv2d_forward_3x3() {
    // 1 batch, 1 ch, 3x3 input, 1 out_ch, 3x3 kernel (all ones), no pad, stride 1
    const int B = 1, IC = 1, OC = 1, H = 3, W = 3, KH = 3, KW = 3;
    std::vector<float> input(B * IC * H * W, 1.0f);
    std::vector<float> weight(OC * IC * KH * KW, 1.0f);

    float* din = to_device(input);
    float* dw = to_device(weight);

    int out_h = 1, out_w = 1;
    float* dout = static_cast<float*>(device_malloc(B * OC * out_h * out_w * sizeof(float)));

    kernels::launch_conv2d_forward_kernel<float>(
        din, dw, nullptr, dout, B, IC, OC, H, W, KH, KW, 1, 1, 0, 0);
    CUDEEP_CHECK_CUDA(cudaDeviceSynchronize());

    auto out = to_host(dout, 1);
    assert(fabs(out[0] - 9.0f) < EPS);

    device_free(din); device_free(dw); device_free(dout);
    printf("[PASS] test_conv2d_forward_3x3\n");
}

void test_conv2d_forward_with_bias() {
    const int B = 1, IC = 1, OC = 1, H = 3, W = 3, KH = 3, KW = 3;
    std::vector<float> input(B * IC * H * W, 1.0f);
    std::vector<float> weight(OC * IC * KH * KW, 1.0f);
    std::vector<float> bias = {10.0f};

    float* din = to_device(input);
    float* dw = to_device(weight);
    float* db = to_device(bias);

    int out_h = 1, out_w = 1;
    float* dout = static_cast<float*>(device_malloc(sizeof(float)));

    kernels::launch_conv2d_forward_kernel<float>(
        din, dw, db, dout, B, IC, OC, H, W, KH, KW, 1, 1, 0, 0);
    CUDEEP_CHECK_CUDA(cudaDeviceSynchronize());

    auto out = to_host(dout, 1);
    assert(fabs(out[0] - 19.0f) < EPS);  // 9 + 10

    device_free(din); device_free(dw); device_free(db); device_free(dout);
    printf("[PASS] test_conv2d_forward_with_bias\n");
}

void test_conv2d_forward_with_padding() {
    // 1 batch, 1 ch, 3x3 input, 1 out_ch, 3x3 kernel (all ones), pad=1, stride 1
    const int B = 1, IC = 1, OC = 1, H = 3, W = 3, KH = 3, KW = 3;
    std::vector<float> input(B * IC * H * W, 1.0f);
    std::vector<float> weight(OC * IC * KH * KW, 1.0f);

    float* din = to_device(input);
    float* dw = to_device(weight);

    int out_h = 3, out_w = 3;
    float* dout = static_cast<float*>(device_malloc(B * OC * out_h * out_w * sizeof(float)));

    kernels::launch_conv2d_forward_kernel<float>(
        din, dw, nullptr, dout, B, IC, OC, H, W, KH, KW, 1, 1, 1, 1);
    CUDEEP_CHECK_CUDA(cudaDeviceSynchronize());

    auto out = to_host(dout, 9);
    // Center element sees full 3x3 window of 1s = 9
    assert(fabs(out[4] - 9.0f) < EPS);
    // Corner sees 2x2 = 4
    assert(fabs(out[0] - 4.0f) < EPS);
    // Edge center sees 3x2 = 6
    assert(fabs(out[1] - 6.0f) < EPS);

    device_free(din); device_free(dw); device_free(dout);
    printf("[PASS] test_conv2d_forward_with_padding\n");
}

int main() {
    printf("=== Elementwise Kernel Tests ===\n");
    test_add_kernel();
    test_sub_kernel();
    test_mul_kernel();
    test_fill_kernel();
    test_scalar_mul_kernel();

    printf("\n=== Activation Kernel Tests ===\n");
    test_relu_forward();
    test_sigmoid_forward();
    test_gelu_forward();
    test_relu_backward();
    test_leaky_relu_forward();

    printf("\n=== Reduction Kernel Tests ===\n");
    test_sum_kernel();
    test_sum_large();

    printf("\n=== Loss Kernel Tests ===\n");
    test_mse_loss_zero();
    test_mse_loss_nonzero();
    test_softmax_kernel();
    test_softmax_batch();

    printf("\n=== Conv2d Kernel Tests ===\n");
    test_conv2d_forward_identity();
    test_conv2d_forward_3x3();
    test_conv2d_forward_with_bias();
    test_conv2d_forward_with_padding();

    printf("\nAll kernel tests passed!\n");
    return 0;
}
