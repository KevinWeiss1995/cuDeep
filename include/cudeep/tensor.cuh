#pragma once

#include "common.cuh"
#include "error.cuh"

#include <vector>
#include <memory>
#include <numeric>
#include <functional>

namespace cudeep {

class Tensor {
public:
    Tensor();
    Tensor(const std::vector<int64_t>& shape, DType dtype = DType::Float32);
    ~Tensor();

    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;

    void* data() const { return data_; }
    const std::vector<int64_t>& shape() const { return shape_; }
    const std::vector<int64_t>& strides() const { return strides_; }
    int64_t ndim() const { return static_cast<int64_t>(shape_.size()); }
    int64_t numel() const;
    size_t nbytes() const;
    DType dtype() const { return dtype_; }

    static Tensor zeros(const std::vector<int64_t>& shape, DType dtype = DType::Float32);
    static Tensor ones(const std::vector<int64_t>& shape, DType dtype = DType::Float32);
    static Tensor randn(const std::vector<int64_t>& shape, DType dtype = DType::Float32);
    static Tensor from_host(const void* host_data, const std::vector<int64_t>& shape, DType dtype = DType::Float32);

    void to_host(void* host_dst) const;
    Tensor reshape(const std::vector<int64_t>& new_shape) const;
    Tensor transpose(int dim0, int dim1) const;
    Tensor contiguous() const;
    bool is_contiguous() const;

    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;

    Tensor matmul(const Tensor& other) const;

    void fill_(float value);
    void zero_();

    cudaStream_t stream() const { return stream_; }
    void set_stream(cudaStream_t stream) { stream_ = stream; }

private:
    void allocate();
    void deallocate();
    void compute_strides();
    size_t dtype_size() const;

    void* data_ = nullptr;
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;
    DType dtype_ = DType::Float32;
    bool owns_data_ = true;
    cudaStream_t stream_ = nullptr;
};

}  // namespace cudeep
