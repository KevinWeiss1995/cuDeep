#include "cudeep/tensor.cuh"
#include "cudeep/error.cuh"

#include <cassert>
#include <cstdio>
#include <vector>
#include <cmath>

using namespace cudeep;

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

void test_tensor_reshape() {
    Tensor t({2, 3, 4});
    Tensor r = t.reshape({6, 4});
    assert(r.shape()[0] == 6);
    assert(r.shape()[1] == 4);
    assert(r.numel() == 24);
    printf("[PASS] test_tensor_reshape\n");
}

int main() {
    test_tensor_create();
    test_tensor_zeros();
    test_tensor_from_host();
    test_tensor_copy();
    test_tensor_reshape();
    printf("\nAll tensor tests passed!\n");
    return 0;
}
