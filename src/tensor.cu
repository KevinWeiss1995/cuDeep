#include "cudeep/tensor.cuh"
#include "cudeep/memory.cuh"
#include "cudeep/kernels/elementwise.cuh"

#include <algorithm>
#include <curand.h>
#include <curand_kernel.h>

namespace cudeep {

Tensor::Tensor() = default;

Tensor::Tensor(const std::vector<int64_t>& shape, DType dtype)
    : shape_(shape), dtype_(dtype), owns_data_(true) {
    compute_strides();
    allocate();
}

Tensor::~Tensor() {
    if (owns_data_) {
        deallocate();
    }
}

Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_), strides_(other.strides_),
      dtype_(other.dtype_), owns_data_(true), stream_(other.stream_) {
    allocate();
    memcpy_d2d(data_, other.data_, nbytes(), stream_);
}

Tensor::Tensor(Tensor&& other) noexcept
    : data_(other.data_), shape_(std::move(other.shape_)),
      strides_(std::move(other.strides_)), dtype_(other.dtype_),
      owns_data_(other.owns_data_), stream_(other.stream_) {
    other.data_ = nullptr;
    other.owns_data_ = false;
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        if (owns_data_) deallocate();
        shape_ = other.shape_;
        strides_ = other.strides_;
        dtype_ = other.dtype_;
        stream_ = other.stream_;
        owns_data_ = true;
        allocate();
        memcpy_d2d(data_, other.data_, nbytes(), stream_);
    }
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        if (owns_data_) deallocate();
        data_ = other.data_;
        shape_ = std::move(other.shape_);
        strides_ = std::move(other.strides_);
        dtype_ = other.dtype_;
        owns_data_ = other.owns_data_;
        stream_ = other.stream_;
        other.data_ = nullptr;
        other.owns_data_ = false;
    }
    return *this;
}

int64_t Tensor::numel() const {
    if (shape_.empty()) return 0;
    return std::accumulate(shape_.begin(), shape_.end(),
                           int64_t(1), std::multiplies<int64_t>());
}

size_t Tensor::nbytes() const {
    return static_cast<size_t>(numel()) * dtype_size();
}

Tensor Tensor::zeros(const std::vector<int64_t>& shape, DType dtype) {
    Tensor t(shape, dtype);
    CUDEEP_CHECK_CUDA(cudaMemset(t.data_, 0, t.nbytes()));
    return t;
}

Tensor Tensor::ones(const std::vector<int64_t>& shape, DType dtype) {
    Tensor t(shape, dtype);
    t.fill_(1.0f);
    return t;
}

Tensor Tensor::randn(const std::vector<int64_t>& shape, DType dtype) {
    Tensor t(shape, dtype);
    // TODO: implement cuRAND-backed randn
    return t;
}

Tensor Tensor::from_host(const void* host_data, const std::vector<int64_t>& shape, DType dtype) {
    Tensor t(shape, dtype);
    memcpy_h2d(t.data_, host_data, t.nbytes(), t.stream_);
    return t;
}

void Tensor::to_host(void* host_dst) const {
    memcpy_d2h(host_dst, data_, nbytes(), stream_);
}

Tensor Tensor::reshape(const std::vector<int64_t>& new_shape) const {
    Tensor out;
    out.data_ = data_;
    out.shape_ = new_shape;
    out.dtype_ = dtype_;
    out.owns_data_ = false;
    out.stream_ = stream_;
    out.compute_strides();

    CUDEEP_ASSERT(out.numel() == numel(),
                  "reshape: total number of elements must match");
    return out;
}

bool Tensor::is_contiguous() const {
    if (shape_.empty()) return true;
    int64_t expected = 1;
    for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
        if (strides_[i] != expected) return false;
        expected *= shape_[i];
    }
    return true;
}

void Tensor::fill_(float value) {
    // TODO: dispatch templated fill kernel based on dtype_
}

void Tensor::zero_() {
    CUDEEP_CHECK_CUDA(cudaMemsetAsync(data_, 0, nbytes(), stream_));
}

void Tensor::allocate() {
    size_t bytes = nbytes();
    if (bytes > 0) {
        data_ = device_malloc(bytes);
    }
}

void Tensor::deallocate() {
    if (data_) {
        device_free(data_);
        data_ = nullptr;
    }
}

void Tensor::compute_strides() {
    strides_.resize(shape_.size());
    if (shape_.empty()) return;
    strides_.back() = 1;
    for (int i = static_cast<int>(shape_.size()) - 2; i >= 0; --i) {
        strides_[i] = strides_[i + 1] * shape_[i + 1];
    }
}

size_t Tensor::dtype_size() const {
    switch (dtype_) {
        case DType::Float16: return 2;
        case DType::Float32: return 4;
        case DType::Float64: return 8;
        default: return 4;
    }
}

}  // namespace cudeep
