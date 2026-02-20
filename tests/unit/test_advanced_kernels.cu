#include "cudeep/tensor.cuh"
#include "cudeep/memory.cuh"
#include "cudeep/error.cuh"
#include "cudeep/kernels/reduce.cuh"
#include "cudeep/kernels/loss.cuh"
#include "cudeep/kernels/conv.cuh"
#include "cudeep/kernels/optim.cuh"
#include "cudeep/kernels/pool.cuh"
#include "cudeep/kernels/norm.cuh"

#include <cassert>
#include <cstdio>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>

using namespace cudeep;

static constexpr float EPS = 1e-3f;

static float* to_device(const std::vector<float>& v) {
    float* d = static_cast<float*>(device_malloc(v.size() * sizeof(float)));
    memcpy_h2d(d, v.data(), v.size() * sizeof(float));
    return d;
}

static int* to_device_int(const std::vector<int>& v) {
    int* d = static_cast<int*>(device_malloc(v.size() * sizeof(int)));
    memcpy_h2d(d, v.data(), v.size() * sizeof(int));
    return d;
}

static std::vector<float> to_host(const float* d, size_t n) {
    std::vector<float> h(n);
    memcpy_d2h(h.data(), d, n * sizeof(float));
    return h;
}

// ---- Reduction tests ----

void test_max_kernel() {
    std::vector<float> data = {3, 1, 4, 1, 5, 9, 2, 6};
    float* din = to_device(data);
    float* dout = static_cast<float*>(device_malloc(sizeof(float)));

    kernels::launch_max_kernel<float>(din, dout, 8);
    CUDEEP_CHECK_CUDA(cudaDeviceSynchronize());

    auto out = to_host(dout, 1);
    assert(fabs(out[0] - 9.0f) < EPS);

    device_free(din); device_free(dout);
    printf("[PASS] test_max_kernel\n");
}

void test_max_kernel_negative() {
    std::vector<float> data = {-5, -3, -1, -7, -2};
    float* din = to_device(data);
    float* dout = static_cast<float*>(device_malloc(sizeof(float)));

    kernels::launch_max_kernel<float>(din, dout, 5);
    CUDEEP_CHECK_CUDA(cudaDeviceSynchronize());

    auto out = to_host(dout, 1);
    assert(fabs(out[0] - (-1.0f)) < EPS);

    device_free(din); device_free(dout);
    printf("[PASS] test_max_kernel_negative\n");
}

void test_max_kernel_large() {
    const int N = 10000;
    std::vector<float> data(N);
    for (int i = 0; i < N; ++i) data[i] = static_cast<float>(i);
    data[7777] = 99999.0f;

    float* din = to_device(data);
    float* dout = static_cast<float*>(device_malloc(sizeof(float)));

    kernels::launch_max_kernel<float>(din, dout, N);
    CUDEEP_CHECK_CUDA(cudaDeviceSynchronize());

    auto out = to_host(dout, 1);
    assert(fabs(out[0] - 99999.0f) < 1.0f);

    device_free(din); device_free(dout);
    printf("[PASS] test_max_kernel_large\n");
}

void test_min_kernel() {
    std::vector<float> data = {3, 1, 4, 1, 5, 9};
    float* din = to_device(data);
    float* dout = static_cast<float*>(device_malloc(sizeof(float)));

    kernels::launch_min_kernel<float>(din, dout, 6);
    CUDEEP_CHECK_CUDA(cudaDeviceSynchronize());

    auto out = to_host(dout, 1);
    assert(fabs(out[0] - 1.0f) < EPS);

    device_free(din); device_free(dout);
    printf("[PASS] test_min_kernel\n");
}

void test_mean_kernel() {
    std::vector<float> data = {2, 4, 6, 8};
    float* din = to_device(data);
    float* dout = static_cast<float*>(device_malloc(sizeof(float)));

    kernels::launch_mean_kernel<float>(din, dout, 4);
    CUDEEP_CHECK_CUDA(cudaDeviceSynchronize());

    auto out = to_host(dout, 1);
    assert(fabs(out[0] - 5.0f) < EPS);

    device_free(din); device_free(dout);
    printf("[PASS] test_mean_kernel\n");
}

void test_sum_along_axis() {
    // Shape [2,3]: [[1,2,3],[4,5,6]]
    // Sum along axis 0 => [5,7,9]
    std::vector<float> data = {1, 2, 3, 4, 5, 6};
    std::vector<int64_t> shape = {2, 3};
    float* din = to_device(data);
    float* dout = static_cast<float*>(device_malloc(3 * sizeof(float)));

    kernels::launch_sum_along_axis_kernel<float>(din, dout, shape.data(), 2, 0);
    CUDEEP_CHECK_CUDA(cudaDeviceSynchronize());

    auto out = to_host(dout, 3);
    assert(fabs(out[0] - 5.0f) < EPS);
    assert(fabs(out[1] - 7.0f) < EPS);
    assert(fabs(out[2] - 9.0f) < EPS);

    device_free(din); device_free(dout);
    printf("[PASS] test_sum_along_axis\n");
}

void test_sum_along_axis_1() {
    // Shape [2,3]: sum along axis 1 => [6, 15]
    std::vector<float> data = {1, 2, 3, 4, 5, 6};
    std::vector<int64_t> shape = {2, 3};
    float* din = to_device(data);
    float* dout = static_cast<float*>(device_malloc(2 * sizeof(float)));

    kernels::launch_sum_along_axis_kernel<float>(din, dout, shape.data(), 2, 1);
    CUDEEP_CHECK_CUDA(cudaDeviceSynchronize());

    auto out = to_host(dout, 2);
    assert(fabs(out[0] - 6.0f) < EPS);
    assert(fabs(out[1] - 15.0f) < EPS);

    device_free(din); device_free(dout);
    printf("[PASS] test_sum_along_axis_1\n");
}

// ---- Cross-entropy loss ----

void test_cross_entropy_perfect() {
    std::vector<float> logits = {100, 0, 0};
    std::vector<int> targets = {0};
    float* dl = to_device(logits);
    int* dt = to_device_int(targets);
    float* dloss = static_cast<float*>(device_malloc(sizeof(float)));

    kernels::launch_cross_entropy_loss_kernel<float>(dl, dt, dloss, 1, 3);
    CUDEEP_CHECK_CUDA(cudaDeviceSynchronize());

    auto loss = to_host(dloss, 1);
    assert(loss[0] < 0.01f);

    device_free(dl); device_free(dt); device_free(dloss);
    printf("[PASS] test_cross_entropy_perfect\n");
}

void test_cross_entropy_uniform() {
    std::vector<float> logits = {0, 0, 0};
    std::vector<int> targets = {1};
    float* dl = to_device(logits);
    int* dt = to_device_int(targets);
    float* dloss = static_cast<float*>(device_malloc(sizeof(float)));

    kernels::launch_cross_entropy_loss_kernel<float>(dl, dt, dloss, 1, 3);
    CUDEEP_CHECK_CUDA(cudaDeviceSynchronize());

    auto loss = to_host(dloss, 1);
    float expected = logf(3.0f);
    assert(fabs(loss[0] - expected) < 0.01f);

    device_free(dl); device_free(dt); device_free(dloss);
    printf("[PASS] test_cross_entropy_uniform\n");
}

// ---- Conv2d backward ----

void test_conv2d_backward_data() {
    // 1 batch, 1 in_ch, 3x3 input, 1 out_ch, 1x1 kernel (weight=2), no pad, stride 1
    // grad_output is all 1s (3x3), weight is 2 => grad_input should be all 2s
    const int B = 1, IC = 1, OC = 1, H = 3, W = 3, KH = 1, KW = 1;
    std::vector<float> grad_output(B * OC * H * W, 1.0f);
    std::vector<float> weight = {2.0f};

    float* dgo = to_device(grad_output);
    float* dw = to_device(weight);
    float* dgi = static_cast<float*>(device_malloc(B * IC * H * W * sizeof(float)));

    kernels::launch_conv2d_backward_data_kernel<float>(
        dgo, dw, dgi, B, IC, OC, H, W, KH, KW, 1, 1, 0, 0);
    CUDEEP_CHECK_CUDA(cudaDeviceSynchronize());

    auto gi = to_host(dgi, B * IC * H * W);
    for (int i = 0; i < B * IC * H * W; ++i) {
        assert(fabs(gi[i] - 2.0f) < EPS);
    }

    device_free(dgo); device_free(dw); device_free(dgi);
    printf("[PASS] test_conv2d_backward_data\n");
}

void test_conv2d_backward_weight() {
    // 1 batch, 1 in_ch, 3x3 input (all 1s), 1 out_ch, 1x1 kernel, no pad, stride 1
    // grad_output is all 1s => grad_weight = sum of input = 9
    const int B = 1, IC = 1, OC = 1, H = 3, W = 3, KH = 1, KW = 1;
    std::vector<float> grad_output(B * OC * H * W, 1.0f);
    std::vector<float> input(B * IC * H * W, 1.0f);

    float* dgo = to_device(grad_output);
    float* din = to_device(input);
    float* dgw = static_cast<float*>(device_malloc(OC * IC * KH * KW * sizeof(float)));

    kernels::launch_conv2d_backward_weight_kernel<float>(
        dgo, din, dgw, B, IC, OC, H, W, KH, KW, 1, 1, 0, 0);
    CUDEEP_CHECK_CUDA(cudaDeviceSynchronize());

    auto gw = to_host(dgw, 1);
    assert(fabs(gw[0] - 9.0f) < EPS);

    device_free(dgo); device_free(din); device_free(dgw);
    printf("[PASS] test_conv2d_backward_weight\n");
}

// ---- Optimizer kernels ----

void test_sgd_basic() {
    std::vector<float> param = {10.0f, 20.0f};
    std::vector<float> grad = {1.0f, 2.0f};
    float* dp = to_device(param);
    float* dg = to_device(grad);

    kernels::launch_sgd_update_kernel<float>(dp, dg, nullptr, 2, 0.1f, 0.0f, 0.0f);
    CUDEEP_CHECK_CUDA(cudaDeviceSynchronize());

    auto result = to_host(dp, 2);
    assert(fabs(result[0] - 9.9f) < EPS);
    assert(fabs(result[1] - 19.8f) < EPS);

    device_free(dp); device_free(dg);
    printf("[PASS] test_sgd_basic\n");
}

void test_adam_basic() {
    std::vector<float> param = {10.0f};
    std::vector<float> grad = {1.0f};
    std::vector<float> m_vec = {0.0f};
    std::vector<float> v_vec = {0.0f};
    float* dp = to_device(param);
    float* dg = to_device(grad);
    float* dm = to_device(m_vec);
    float* dv = to_device(v_vec);

    float initial = param[0];
    kernels::launch_adam_update_kernel<float>(
        dp, dg, dm, dv, 1, 0.01f, 0.9f, 0.999f, 1e-8f, 0.0f, 1);
    CUDEEP_CHECK_CUDA(cudaDeviceSynchronize());

    auto result = to_host(dp, 1);
    assert(result[0] < initial);

    device_free(dp); device_free(dg); device_free(dm); device_free(dv);
    printf("[PASS] test_adam_basic\n");
}

// ---- Pooling kernels ----

void test_maxpool2d() {
    // 1 batch, 1 ch, 4x4, kernel 2x2, stride 2 => 2x2
    std::vector<float> input(16);
    std::iota(input.begin(), input.end(), 1.0f);
    float* din = to_device(input);

    float* dout = static_cast<float*>(device_malloc(4 * sizeof(float)));
    kernels::launch_maxpool2d_forward_kernel<float>(
        din, dout, 1, 1, 4, 4, 2, 2, 2, 2, 0, 0);
    CUDEEP_CHECK_CUDA(cudaDeviceSynchronize());

    auto out = to_host(dout, 4);
    assert(fabs(out[0] - 6.0f) < EPS);
    assert(fabs(out[1] - 8.0f) < EPS);
    assert(fabs(out[2] - 14.0f) < EPS);
    assert(fabs(out[3] - 16.0f) < EPS);

    device_free(din); device_free(dout);
    printf("[PASS] test_maxpool2d\n");
}

void test_avgpool2d() {
    std::vector<float> input(16, 1.0f);
    float* din = to_device(input);

    float* dout = static_cast<float*>(device_malloc(4 * sizeof(float)));
    kernels::launch_avgpool2d_forward_kernel<float>(
        din, dout, 1, 1, 4, 4, 2, 2, 2, 2, 0, 0);
    CUDEEP_CHECK_CUDA(cudaDeviceSynchronize());

    auto out = to_host(dout, 4);
    for (int i = 0; i < 4; ++i) {
        assert(fabs(out[i] - 1.0f) < EPS);
    }

    device_free(din); device_free(dout);
    printf("[PASS] test_avgpool2d\n");
}

// ---- Normalization kernels ----

void test_layernorm() {
    // 2 batch, normalized_size=4
    // Input: [1,2,3,4, 5,6,7,8] with weight=1, bias=0
    std::vector<float> input = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> weight = {1, 1, 1, 1};
    std::vector<float> bias = {0, 0, 0, 0};

    float* din = to_device(input);
    float* dw = to_device(weight);
    float* db = to_device(bias);
    float* dout = static_cast<float*>(device_malloc(8 * sizeof(float)));

    kernels::launch_layernorm_forward_kernel<float>(
        din, dout, dw, db, 2, 4, 1e-5f);
    CUDEEP_CHECK_CUDA(cudaDeviceSynchronize());

    auto out = to_host(dout, 8);
    // Each row should have mean ~0 and std ~1
    float sum1 = out[0] + out[1] + out[2] + out[3];
    float sum2 = out[4] + out[5] + out[6] + out[7];
    assert(fabs(sum1) < 0.01f);
    assert(fabs(sum2) < 0.01f);

    device_free(din); device_free(dw); device_free(db); device_free(dout);
    printf("[PASS] test_layernorm\n");
}

int main() {
    printf("=== Reduction Kernel Tests ===\n");
    test_max_kernel();
    test_max_kernel_negative();
    test_max_kernel_large();
    test_min_kernel();
    test_mean_kernel();
    test_sum_along_axis();
    test_sum_along_axis_1();

    printf("\n=== Cross-Entropy Loss Tests ===\n");
    test_cross_entropy_perfect();
    test_cross_entropy_uniform();

    printf("\n=== Conv2d Backward Tests ===\n");
    test_conv2d_backward_data();
    test_conv2d_backward_weight();

    printf("\n=== Optimizer Kernel Tests ===\n");
    test_sgd_basic();
    test_adam_basic();

    printf("\n=== Pooling Kernel Tests ===\n");
    test_maxpool2d();
    test_avgpool2d();

    printf("\n=== Normalization Kernel Tests ===\n");
    test_layernorm();

    printf("\nAll advanced kernel tests passed!\n");
    return 0;
}
