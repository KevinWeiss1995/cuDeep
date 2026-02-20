#include "cudeep/tensor.cuh"
#include "cudeep/memory.cuh"
#include "cudeep/kernels/elementwise.cuh"
#include "cudeep/kernels/matmul.cuh"

#include <algorithm>
#include <chrono>
#include <curand.h>

namespace cudeep {

namespace {

#define CUDEEP_CHECK_CURAND(call)                                             \
    do {                                                                      \
        curandStatus_t status = (call);                                       \
        if (status != CURAND_STATUS_SUCCESS) {                                \
            throw std::runtime_error(                                         \
                std::string("cuRAND error at ") + __FILE__ + ":" +            \
                std::to_string(__LINE__) + " — code " +                       \
                std::to_string(static_cast<int>(status)));                     \
        }                                                                     \
    } while (0)

template <typename T>
__global__ void strided_copy_kernel(
    const T* __restrict__ src, T* __restrict__ dst,
    const int64_t* __restrict__ shape,
    const int64_t* __restrict__ src_strides,
    int ndim, int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    int64_t src_offset = 0;
    int64_t remaining = idx;
    for (int d = 0; d < ndim; ++d) {
        int64_t stride_after = 1;
        for (int dd = d + 1; dd < ndim; ++dd) stride_after *= shape[dd];
        int64_t coord = remaining / stride_after;
        remaining %= stride_after;
        src_offset += coord * src_strides[d];
    }

    dst[idx] = src[src_offset];
}

template <typename T>
void launch_strided_copy(const T* src, T* dst,
                         const std::vector<int64_t>& shape,
                         const std::vector<int64_t>& src_strides,
                         int64_t numel, cudaStream_t stream) {
    int ndim = static_cast<int>(shape.size());

    int64_t* d_shape = static_cast<int64_t*>(device_malloc(ndim * sizeof(int64_t)));
    int64_t* d_strides = static_cast<int64_t*>(device_malloc(ndim * sizeof(int64_t)));
    memcpy_h2d(d_shape, shape.data(), ndim * sizeof(int64_t), stream);
    memcpy_h2d(d_strides, src_strides.data(), ndim * sizeof(int64_t), stream);

    int threads = DEFAULT_BLOCK_SIZE;
    int blocks = ceil_div(static_cast<int>(numel), threads);
    strided_copy_kernel<<<blocks, threads, 0, stream>>>(
        src, dst, d_shape, d_strides, ndim, numel);
    CUDEEP_CHECK_LAST_KERNEL();

    CUDEEP_CHECK_CUDA(cudaStreamSynchronize(stream));
    device_free(d_shape);
    device_free(d_strides);
}

}  // anonymous namespace

// --- Constructors / destructors / assignment ---

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

// --- Metadata ---

int64_t Tensor::numel() const {
    if (shape_.empty()) return 0;
    return std::accumulate(shape_.begin(), shape_.end(),
                           int64_t(1), std::multiplies<int64_t>());
}

size_t Tensor::nbytes() const {
    return static_cast<size_t>(numel()) * dtype_size();
}

// --- Static factory methods ---

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
    int64_t n = t.numel();
    if (n == 0) return t;

    curandGenerator_t gen;
    CUDEEP_CHECK_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));

    auto seed = static_cast<unsigned long long>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    CUDEEP_CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, seed));

    int64_t gen_count = n + (n % 2);  // curandGenerate*Normal requires even count

    switch (dtype) {
        case DType::Float32: {
            if (gen_count == n) {
                CUDEEP_CHECK_CURAND(curandGenerateNormal(
                    gen, static_cast<float*>(t.data_), gen_count, 0.0f, 1.0f));
            } else {
                void* tmp = device_malloc(gen_count * sizeof(float));
                CUDEEP_CHECK_CURAND(curandGenerateNormal(
                    gen, static_cast<float*>(tmp), gen_count, 0.0f, 1.0f));
                memcpy_d2d(t.data_, tmp, n * sizeof(float));
                device_free(tmp);
            }
            break;
        }
        case DType::Float64: {
            if (gen_count == n) {
                CUDEEP_CHECK_CURAND(curandGenerateNormalDouble(
                    gen, static_cast<double*>(t.data_), gen_count, 0.0, 1.0));
            } else {
                void* tmp = device_malloc(gen_count * sizeof(double));
                CUDEEP_CHECK_CURAND(curandGenerateNormalDouble(
                    gen, static_cast<double*>(tmp), gen_count, 0.0, 1.0));
                memcpy_d2d(t.data_, tmp, n * sizeof(double));
                device_free(tmp);
            }
            break;
        }
        default:
            curandDestroyGenerator(gen);
            throw std::runtime_error("randn: unsupported dtype (float16 not yet supported)");
    }

    CUDEEP_CHECK_CURAND(curandDestroyGenerator(gen));
    return t;
}

Tensor Tensor::from_host(const void* host_data, const std::vector<int64_t>& shape, DType dtype) {
    Tensor t(shape, dtype);
    memcpy_h2d(t.data_, host_data, t.nbytes(), t.stream_);
    return t;
}

// --- Data movement ---

void Tensor::to_host(void* host_dst) const {
    if (!is_contiguous()) {
        Tensor tmp = contiguous();
        memcpy_d2h(host_dst, tmp.data_, tmp.nbytes(), tmp.stream_);
        return;
    }
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

Tensor Tensor::transpose(int dim0, int dim1) const {
    CUDEEP_ASSERT(dim0 >= 0 && dim0 < ndim(), "transpose: dim0 out of range");
    CUDEEP_ASSERT(dim1 >= 0 && dim1 < ndim(), "transpose: dim1 out of range");

    Tensor out;
    out.data_ = data_;
    out.shape_ = shape_;
    out.strides_ = strides_;
    out.dtype_ = dtype_;
    out.owns_data_ = false;
    out.stream_ = stream_;

    std::swap(out.shape_[dim0], out.shape_[dim1]);
    std::swap(out.strides_[dim0], out.strides_[dim1]);
    return out;
}

Tensor Tensor::contiguous() const {
    if (is_contiguous()) {
        return Tensor(*this);
    }

    Tensor out(shape_, dtype_);
    out.stream_ = stream_;

    switch (dtype_) {
        case DType::Float32:
            launch_strided_copy(
                static_cast<const float*>(data_),
                static_cast<float*>(out.data_),
                shape_, strides_, numel(), stream_);
            break;
        case DType::Float64:
            launch_strided_copy(
                static_cast<const double*>(data_),
                static_cast<double*>(out.data_),
                shape_, strides_, numel(), stream_);
            break;
        default:
            throw std::runtime_error("contiguous: unsupported dtype");
    }
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

// --- Elementwise operators ---

Tensor Tensor::operator+(const Tensor& other) const {
    CUDEEP_ASSERT(shape_ == other.shape_, "operator+: shapes must match");
    CUDEEP_ASSERT(dtype_ == other.dtype_, "operator+: dtypes must match");
    CUDEEP_ASSERT(is_contiguous() && other.is_contiguous(),
                  "operator+: both tensors must be contiguous (call .contiguous() first)");

    Tensor out(shape_, dtype_);
    out.stream_ = stream_;
    int64_t n = numel();

    switch (dtype_) {
        case DType::Float32:
            kernels::launch_add_kernel(
                static_cast<const float*>(data_),
                static_cast<const float*>(other.data_),
                static_cast<float*>(out.data_), n, stream_);
            break;
        case DType::Float64:
            kernels::launch_add_kernel(
                static_cast<const double*>(data_),
                static_cast<const double*>(other.data_),
                static_cast<double*>(out.data_), n, stream_);
            break;
        default:
            throw std::runtime_error("operator+: unsupported dtype");
    }
    return out;
}

Tensor Tensor::operator-(const Tensor& other) const {
    CUDEEP_ASSERT(shape_ == other.shape_, "operator-: shapes must match");
    CUDEEP_ASSERT(dtype_ == other.dtype_, "operator-: dtypes must match");
    CUDEEP_ASSERT(is_contiguous() && other.is_contiguous(),
                  "operator-: both tensors must be contiguous (call .contiguous() first)");

    Tensor out(shape_, dtype_);
    out.stream_ = stream_;
    int64_t n = numel();

    switch (dtype_) {
        case DType::Float32:
            kernels::launch_sub_kernel(
                static_cast<const float*>(data_),
                static_cast<const float*>(other.data_),
                static_cast<float*>(out.data_), n, stream_);
            break;
        case DType::Float64:
            kernels::launch_sub_kernel(
                static_cast<const double*>(data_),
                static_cast<const double*>(other.data_),
                static_cast<double*>(out.data_), n, stream_);
            break;
        default:
            throw std::runtime_error("operator-: unsupported dtype");
    }
    return out;
}

Tensor Tensor::operator*(const Tensor& other) const {
    CUDEEP_ASSERT(shape_ == other.shape_, "operator*: shapes must match");
    CUDEEP_ASSERT(dtype_ == other.dtype_, "operator*: dtypes must match");
    CUDEEP_ASSERT(is_contiguous() && other.is_contiguous(),
                  "operator*: both tensors must be contiguous (call .contiguous() first)");

    Tensor out(shape_, dtype_);
    out.stream_ = stream_;
    int64_t n = numel();

    switch (dtype_) {
        case DType::Float32:
            kernels::launch_mul_kernel(
                static_cast<const float*>(data_),
                static_cast<const float*>(other.data_),
                static_cast<float*>(out.data_), n, stream_);
            break;
        case DType::Float64:
            kernels::launch_mul_kernel(
                static_cast<const double*>(data_),
                static_cast<const double*>(other.data_),
                static_cast<double*>(out.data_), n, stream_);
            break;
        default:
            throw std::runtime_error("operator*: unsupported dtype");
    }
    return out;
}

// --- Matrix multiplication ---

Tensor Tensor::matmul(const Tensor& other) const {
    CUDEEP_ASSERT(ndim() == 2 && other.ndim() == 2,
                  "matmul: tensors must be 2D");
    CUDEEP_ASSERT(shape_[1] == other.shape_[0],
                  "matmul: inner dimensions must match (A is MxK, B is KxN)");
    CUDEEP_ASSERT(dtype_ == other.dtype_,
                  "matmul: dtypes must match");
    CUDEEP_ASSERT(is_contiguous() && other.is_contiguous(),
                  "matmul: both tensors must be contiguous (call .contiguous() first)");

    int M = static_cast<int>(shape_[0]);
    int K = static_cast<int>(shape_[1]);
    int N = static_cast<int>(other.shape_[1]);

    Tensor out({M, N}, dtype_);
    out.stream_ = stream_;

    switch (dtype_) {
        case DType::Float32:
            kernels::launch_matmul_kernel(
                static_cast<const float*>(data_),
                static_cast<const float*>(other.data_),
                static_cast<float*>(out.data_),
                M, N, K, stream_);
            break;
        case DType::Float64:
            kernels::launch_matmul_kernel(
                static_cast<const double*>(data_),
                static_cast<const double*>(other.data_),
                static_cast<double*>(out.data_),
                M, N, K, stream_);
            break;
        default:
            throw std::runtime_error("matmul: unsupported dtype");
    }
    return out;
}

// --- In-place operations ---

void Tensor::fill_(float value) {
    int64_t n = numel();
    if (n == 0) return;

    switch (dtype_) {
        case DType::Float32:
            kernels::launch_fill_kernel(
                static_cast<float*>(data_), value, n, stream_);
            break;
        case DType::Float64:
            kernels::launch_fill_kernel(
                static_cast<double*>(data_), static_cast<double>(value), n, stream_);
            break;
        default:
            throw std::runtime_error("fill_: unsupported dtype");
    }
}

void Tensor::zero_() {
    CUDEEP_CHECK_CUDA(cudaMemsetAsync(data_, 0, nbytes(), stream_));
}

// --- Private helpers ---

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
