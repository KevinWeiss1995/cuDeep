#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "cudeep/tensor.cuh"
#include "cudeep/memory.cuh"
#include "cudeep/stream.cuh"
#include "cudeep/common.cuh"
#include "cudeep/kernels/activation.cuh"
#include "cudeep/kernels/reduce.cuh"
#include "cudeep/kernels/loss.cuh"
#include "cudeep/kernels/conv.cuh"
#include "cudeep/kernels/pool.cuh"
#include "cudeep/kernels/norm.cuh"
#include "cudeep/kernels/optim.cuh"
#include "cudeep/kernels/elementwise.cuh"
#include "cudeep/kernels/unary.cuh"
#include "cudeep/kernels/matmul.cuh"

namespace py = pybind11;

using namespace cudeep;

namespace {

inline Tensor ensure_contiguous(const Tensor& t) {
    return t.is_contiguous() ? t : t.contiguous();
}

Tensor activation_dispatch(const Tensor& t, kernels::ActivationType act, float alpha = 0.01f) {
    Tensor input = ensure_contiguous(t);
    Tensor out(input.shape(), input.dtype());
    int64_t n = input.numel();
    switch (input.dtype()) {
        case DType::Float32:
            kernels::launch_activation_forward_kernel<float>(
                static_cast<const float*>(input.data()),
                static_cast<float*>(out.data()), n, act, alpha, input.stream());
            break;
        case DType::Float64:
            kernels::launch_activation_forward_kernel<double>(
                static_cast<const double*>(input.data()),
                static_cast<double*>(out.data()), n, act, alpha, input.stream());
            break;
        default:
            throw std::runtime_error("Unsupported dtype for activation");
    }
    return out;
}

Tensor reduce_dispatch(
    const Tensor& t,
    void (*f32_fn)(const float*, float*, int64_t, cudaStream_t),
    void (*f64_fn)(const double*, double*, int64_t, cudaStream_t)
) {
    Tensor input = ensure_contiguous(t);
    Tensor out({1}, input.dtype());
    switch (input.dtype()) {
        case DType::Float32:
            f32_fn(static_cast<const float*>(input.data()),
                   static_cast<float*>(out.data()), input.numel(), input.stream());
            break;
        case DType::Float64:
            f64_fn(static_cast<const double*>(input.data()),
                   static_cast<double*>(out.data()), input.numel(), input.stream());
            break;
        default:
            throw std::runtime_error("Unsupported dtype for reduction");
    }
    return out;
}

}  // anonymous namespace

PYBIND11_MODULE(_cudeep_core, m) {
    m.doc() = "cuDeep: Ultra-high performance deep learning library";

    // ---- Enums ----

    py::enum_<DType>(m, "DType")
        .value("float16", DType::Float16)
        .value("float32", DType::Float32)
        .value("float64", DType::Float64);

    py::enum_<Layout>(m, "Layout")
        .value("NCHW", Layout::NCHW)
        .value("NHWC", Layout::NHWC);

    // ---- Tensor ----

    py::class_<Tensor>(m, "Tensor")
        .def(py::init<const std::vector<int64_t>&, DType>(),
             py::arg("shape"), py::arg("dtype") = DType::Float32)
        .def("shape", &Tensor::shape)
        .def("strides", &Tensor::strides)
        .def("ndim", &Tensor::ndim)
        .def("numel", &Tensor::numel)
        .def("nbytes", &Tensor::nbytes)
        .def("dtype", &Tensor::dtype)
        .def("is_contiguous", &Tensor::is_contiguous)
        .def("reshape", &Tensor::reshape)
        .def("transpose", &Tensor::transpose, py::arg("dim0"), py::arg("dim1"))
        .def("contiguous", &Tensor::contiguous)
        .def("fill_", &Tensor::fill_)
        .def("zero_", &Tensor::zero_)
        .def("matmul", &Tensor::matmul)
        .def("__add__", &Tensor::operator+)
        .def("__sub__", &Tensor::operator-)
        .def("__mul__", &Tensor::operator*)
        .def_static("zeros", &Tensor::zeros,
                     py::arg("shape"), py::arg("dtype") = DType::Float32)
        .def_static("ones", &Tensor::ones,
                     py::arg("shape"), py::arg("dtype") = DType::Float32)
        .def_static("randn", &Tensor::randn,
                     py::arg("shape"), py::arg("dtype") = DType::Float32)
        .def("numpy", [](const Tensor& t) -> py::object {
            auto dtype = t.dtype();
            auto shape = t.shape();
            std::vector<ssize_t> py_shape(shape.begin(), shape.end());

            if (dtype == DType::Float32) {
                py::array_t<float> result(py_shape);
                t.to_host(result.mutable_data());
                return std::move(result);
            } else if (dtype == DType::Float64) {
                py::array_t<double> result(py_shape);
                t.to_host(result.mutable_data());
                return std::move(result);
            }
            throw std::runtime_error("numpy() unsupported for this dtype");
        })
        .def_static("from_numpy", [](py::array arr) {
            py::buffer_info buf = arr.request();
            std::vector<int64_t> shape(buf.shape.begin(), buf.shape.end());

            DType dtype;
            if (buf.format == py::format_descriptor<float>::format()) {
                dtype = DType::Float32;
            } else if (buf.format == py::format_descriptor<double>::format()) {
                dtype = DType::Float64;
            } else {
                throw std::runtime_error("Unsupported numpy dtype");
            }

            return Tensor::from_host(buf.ptr, shape, dtype);
        })
        .def("__repr__", [](const Tensor& t) {
            std::string s = "cuDeep.Tensor(shape=[";
            auto& shape = t.shape();
            for (size_t i = 0; i < shape.size(); ++i) {
                if (i > 0) s += ", ";
                s += std::to_string(shape[i]);
            }
            s += "], dtype=";
            switch (t.dtype()) {
                case DType::Float16: s += "float16"; break;
                case DType::Float32: s += "float32"; break;
                case DType::Float64: s += "float64"; break;
            }
            s += ")";
            return s;
        });

    // ---- Stream / Event / MemoryPool ----

    py::class_<Stream>(m, "Stream")
        .def(py::init<>())
        .def("synchronize", &Stream::synchronize);

    py::class_<Event>(m, "Event")
        .def(py::init<>())
        .def("record", [](Event& e) { e.record(nullptr); })
        .def("record", [](Event& e, Stream& s) { e.record(s.get()); }, py::arg("stream"))
        .def("synchronize", &Event::synchronize)
        .def("elapsed_ms", &Event::elapsed_ms);

    py::class_<MemoryPool, std::unique_ptr<MemoryPool, py::nodelete>>(m, "MemoryPool")
        .def_static("instance", &MemoryPool::instance, py::return_value_policy::reference)
        .def("release_cached", &MemoryPool::release_cached)
        .def_property_readonly("allocated_bytes", &MemoryPool::allocated_bytes)
        .def_property_readonly("cached_bytes", &MemoryPool::cached_bytes);

    // ---- Functional: Activations ----

    m.def("relu", [](const Tensor& t) {
        return activation_dispatch(t, kernels::ActivationType::ReLU);
    }, py::arg("input"));

    m.def("sigmoid", [](const Tensor& t) {
        return activation_dispatch(t, kernels::ActivationType::Sigmoid);
    }, py::arg("input"));

    m.def("tanh_act", [](const Tensor& t) {
        return activation_dispatch(t, kernels::ActivationType::Tanh);
    }, py::arg("input"));

    m.def("gelu", [](const Tensor& t) {
        return activation_dispatch(t, kernels::ActivationType::GELU);
    }, py::arg("input"));

    m.def("silu", [](const Tensor& t) {
        return activation_dispatch(t, kernels::ActivationType::SiLU);
    }, py::arg("input"));

    m.def("leaky_relu", [](const Tensor& t, float alpha) {
        return activation_dispatch(t, kernels::ActivationType::LeakyReLU, alpha);
    }, py::arg("input"), py::arg("alpha") = 0.01f);

    // ---- Functional: Reductions ----

    m.def("sum", [](const Tensor& t) {
        return reduce_dispatch(t,
            kernels::launch_sum_kernel<float>,
            kernels::launch_sum_kernel<double>);
    }, py::arg("input"));

    m.def("mean", [](const Tensor& t) {
        return reduce_dispatch(t,
            kernels::launch_mean_kernel<float>,
            kernels::launch_mean_kernel<double>);
    }, py::arg("input"));

    m.def("max", [](const Tensor& t) {
        return reduce_dispatch(t,
            kernels::launch_max_kernel<float>,
            kernels::launch_max_kernel<double>);
    }, py::arg("input"));

    m.def("min", [](const Tensor& t) {
        return reduce_dispatch(t,
            kernels::launch_min_kernel<float>,
            kernels::launch_min_kernel<double>);
    }, py::arg("input"));

    // ---- Functional: Softmax ----

    m.def("softmax", [](const Tensor& t, int dim) {
        Tensor input = ensure_contiguous(t);
        auto& shape = input.shape();
        if (input.ndim() != 2 || dim != 1) {
            throw std::runtime_error("softmax: currently supports 2D tensors with dim=1");
        }
        int batch = static_cast<int>(shape[0]);
        int classes = static_cast<int>(shape[1]);
        Tensor out(shape, input.dtype());
        switch (input.dtype()) {
            case DType::Float32:
                kernels::launch_softmax_kernel<float>(
                    static_cast<const float*>(input.data()),
                    static_cast<float*>(out.data()), batch, classes, input.stream());
                break;
            case DType::Float64:
                kernels::launch_softmax_kernel<double>(
                    static_cast<const double*>(input.data()),
                    static_cast<double*>(out.data()), batch, classes, input.stream());
                break;
            default:
                throw std::runtime_error("softmax: unsupported dtype");
        }
        return out;
    }, py::arg("input"), py::arg("dim") = 1);

    // ---- Functional: Loss ----

    m.def("mse_loss", [](const Tensor& pred, const Tensor& target) {
        Tensor p = ensure_contiguous(pred);
        Tensor t = ensure_contiguous(target);
        Tensor loss({1}, p.dtype());
        int64_t n = p.numel();
        switch (p.dtype()) {
            case DType::Float32:
                kernels::launch_mse_loss_kernel<float>(
                    static_cast<const float*>(p.data()),
                    static_cast<const float*>(t.data()),
                    static_cast<float*>(loss.data()), n, p.stream());
                break;
            case DType::Float64:
                kernels::launch_mse_loss_kernel<double>(
                    static_cast<const double*>(p.data()),
                    static_cast<const double*>(t.data()),
                    static_cast<double*>(loss.data()), n, p.stream());
                break;
            default:
                throw std::runtime_error("mse_loss: unsupported dtype");
        }
        return loss;
    }, py::arg("pred"), py::arg("target"));

    m.def("cross_entropy_loss", [](const Tensor& logits, py::array_t<int> targets) {
        Tensor input = ensure_contiguous(logits);
        auto& shape = input.shape();
        int batch = static_cast<int>(shape[0]);
        int num_classes = static_cast<int>(shape[1]);
        py::buffer_info buf = targets.request();

        int* d_targets = static_cast<int*>(device_malloc(batch * sizeof(int)));
        memcpy_h2d(d_targets, buf.ptr, batch * sizeof(int));

        Tensor loss({1}, input.dtype());
        switch (input.dtype()) {
            case DType::Float32:
                kernels::launch_cross_entropy_loss_kernel<float>(
                    static_cast<const float*>(input.data()),
                    d_targets,
                    static_cast<float*>(loss.data()),
                    batch, num_classes, input.stream());
                break;
            case DType::Float64:
                kernels::launch_cross_entropy_loss_kernel<double>(
                    static_cast<const double*>(input.data()),
                    d_targets,
                    static_cast<double*>(loss.data()),
                    batch, num_classes, input.stream());
                break;
            default:
                device_free(d_targets);
                throw std::runtime_error("cross_entropy_loss: unsupported dtype");
        }
        CUDEEP_CHECK_CUDA(cudaStreamSynchronize(input.stream()));
        device_free(d_targets);
        return loss;
    }, py::arg("logits"), py::arg("targets"));

    // ---- Functional: Conv2d ----

    m.def("conv2d_forward", [](
        const Tensor& input, const Tensor& weight,
        py::object bias_obj,
        std::vector<int> stride, std::vector<int> padding
    ) {
        Tensor inp = ensure_contiguous(input);
        Tensor w = ensure_contiguous(weight);
        auto& ishape = inp.shape();
        auto& wshape = w.shape();
        int B  = static_cast<int>(ishape[0]);
        int IC = static_cast<int>(ishape[1]);
        int IH = static_cast<int>(ishape[2]);
        int IW = static_cast<int>(ishape[3]);
        int OC = static_cast<int>(wshape[0]);
        int KH = static_cast<int>(wshape[2]);
        int KW = static_cast<int>(wshape[3]);
        int SH = stride[0], SW = stride[1];
        int PH = padding[0], PW = padding[1];
        int OH = (IH + 2 * PH - KH) / SH + 1;
        int OW = (IW + 2 * PW - KW) / SW + 1;

        Tensor out({B, OC, OH, OW}, inp.dtype());

        const void* bias_ptr = nullptr;
        Tensor bias_tensor;
        if (!bias_obj.is_none()) {
            bias_tensor = ensure_contiguous(bias_obj.cast<Tensor>());
            bias_ptr = bias_tensor.data();
        }

        switch (inp.dtype()) {
            case DType::Float32:
                kernels::launch_conv2d_forward_kernel<float>(
                    static_cast<const float*>(inp.data()),
                    static_cast<const float*>(w.data()),
                    static_cast<const float*>(bias_ptr),
                    static_cast<float*>(out.data()),
                    B, IC, OC, IH, IW, KH, KW, SH, SW, PH, PW,
                    inp.stream());
                break;
            case DType::Float64:
                kernels::launch_conv2d_forward_kernel<double>(
                    static_cast<const double*>(inp.data()),
                    static_cast<const double*>(w.data()),
                    static_cast<const double*>(bias_ptr),
                    static_cast<double*>(out.data()),
                    B, IC, OC, IH, IW, KH, KW, SH, SW, PH, PW,
                    inp.stream());
                break;
            default:
                throw std::runtime_error("conv2d_forward: unsupported dtype");
        }
        return out;
    }, py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(),
       py::arg("stride") = std::vector<int>{1, 1},
       py::arg("padding") = std::vector<int>{0, 0});

    // ---- Functional: Pooling ----

    m.def("max_pool2d", [](
        const Tensor& input,
        std::vector<int> kernel_size, std::vector<int> stride, std::vector<int> padding
    ) {
        Tensor inp = ensure_contiguous(input);
        auto& shape = inp.shape();
        int B = static_cast<int>(shape[0]), C = static_cast<int>(shape[1]);
        int H = static_cast<int>(shape[2]), W = static_cast<int>(shape[3]);
        int KH = kernel_size[0], KW = kernel_size[1];
        int SH = stride[0], SW = stride[1];
        int PH = padding[0], PW = padding[1];
        int OH = (H + 2 * PH - KH) / SH + 1;
        int OW = (W + 2 * PW - KW) / SW + 1;

        Tensor out({B, C, OH, OW}, inp.dtype());
        switch (inp.dtype()) {
            case DType::Float32:
                kernels::launch_maxpool2d_forward_kernel<float>(
                    static_cast<const float*>(inp.data()),
                    static_cast<float*>(out.data()),
                    B, C, H, W, KH, KW, SH, SW, PH, PW, inp.stream());
                break;
            case DType::Float64:
                kernels::launch_maxpool2d_forward_kernel<double>(
                    static_cast<const double*>(inp.data()),
                    static_cast<double*>(out.data()),
                    B, C, H, W, KH, KW, SH, SW, PH, PW, inp.stream());
                break;
            default:
                throw std::runtime_error("max_pool2d: unsupported dtype");
        }
        return out;
    }, py::arg("input"),
       py::arg("kernel_size"), py::arg("stride"), py::arg("padding") = std::vector<int>{0, 0});

    m.def("avg_pool2d", [](
        const Tensor& input,
        std::vector<int> kernel_size, std::vector<int> stride, std::vector<int> padding
    ) {
        Tensor inp = ensure_contiguous(input);
        auto& shape = inp.shape();
        int B = static_cast<int>(shape[0]), C = static_cast<int>(shape[1]);
        int H = static_cast<int>(shape[2]), W = static_cast<int>(shape[3]);
        int KH = kernel_size[0], KW = kernel_size[1];
        int SH = stride[0], SW = stride[1];
        int PH = padding[0], PW = padding[1];
        int OH = (H + 2 * PH - KH) / SH + 1;
        int OW = (W + 2 * PW - KW) / SW + 1;

        Tensor out({B, C, OH, OW}, inp.dtype());
        switch (inp.dtype()) {
            case DType::Float32:
                kernels::launch_avgpool2d_forward_kernel<float>(
                    static_cast<const float*>(inp.data()),
                    static_cast<float*>(out.data()),
                    B, C, H, W, KH, KW, SH, SW, PH, PW, inp.stream());
                break;
            case DType::Float64:
                kernels::launch_avgpool2d_forward_kernel<double>(
                    static_cast<const double*>(inp.data()),
                    static_cast<double*>(out.data()),
                    B, C, H, W, KH, KW, SH, SW, PH, PW, inp.stream());
                break;
            default:
                throw std::runtime_error("avg_pool2d: unsupported dtype");
        }
        return out;
    }, py::arg("input"),
       py::arg("kernel_size"), py::arg("stride"), py::arg("padding") = std::vector<int>{0, 0});

    // ---- Functional: Normalization ----

    m.def("batchnorm_forward", [](
        const Tensor& input, const Tensor& weight, const Tensor& bias,
        py::object running_mean_obj, py::object running_var_obj,
        float eps, float momentum, bool training
    ) {
        Tensor inp = ensure_contiguous(input);
        auto& shape = inp.shape();
        int B = static_cast<int>(shape[0]), C = static_cast<int>(shape[1]);
        int spatial = 1;
        for (int i = 2; i < inp.ndim(); ++i)
            spatial *= static_cast<int>(shape[i]);

        Tensor out(shape, inp.dtype());

        void* rm_ptr = nullptr;
        void* rv_ptr = nullptr;
        Tensor rm_tensor, rv_tensor;
        if (!running_mean_obj.is_none()) {
            rm_tensor = running_mean_obj.cast<Tensor>();
            rm_ptr = const_cast<void*>(rm_tensor.data());
        }
        if (!running_var_obj.is_none()) {
            rv_tensor = running_var_obj.cast<Tensor>();
            rv_ptr = const_cast<void*>(rv_tensor.data());
        }

        Tensor w = ensure_contiguous(weight);
        Tensor b = ensure_contiguous(bias);
        switch (inp.dtype()) {
            case DType::Float32:
                kernels::launch_batchnorm_forward_kernel<float>(
                    static_cast<const float*>(inp.data()),
                    static_cast<float*>(out.data()),
                    static_cast<const float*>(w.data()),
                    static_cast<const float*>(b.data()),
                    static_cast<float*>(rm_ptr),
                    static_cast<float*>(rv_ptr),
                    B, C, spatial, eps, momentum, training, inp.stream());
                break;
            case DType::Float64:
                kernels::launch_batchnorm_forward_kernel<double>(
                    static_cast<const double*>(inp.data()),
                    static_cast<double*>(out.data()),
                    static_cast<const double*>(w.data()),
                    static_cast<const double*>(b.data()),
                    static_cast<double*>(rm_ptr),
                    static_cast<double*>(rv_ptr),
                    B, C, spatial, eps, momentum, training, inp.stream());
                break;
            default:
                throw std::runtime_error("batchnorm_forward: unsupported dtype");
        }
        return out;
    }, py::arg("input"), py::arg("weight"), py::arg("bias"),
       py::arg("running_mean") = py::none(), py::arg("running_var") = py::none(),
       py::arg("eps") = 1e-5f, py::arg("momentum") = 0.1f, py::arg("training") = true);

    m.def("layernorm_forward", [](
        const Tensor& input, const Tensor& weight, const Tensor& bias,
        int normalized_size, float eps
    ) {
        Tensor inp = ensure_contiguous(input);
        Tensor w = ensure_contiguous(weight);
        Tensor b = ensure_contiguous(bias);
        auto& shape = inp.shape();
        int batch_size = static_cast<int>(inp.numel()) / normalized_size;
        Tensor out(shape, inp.dtype());
        switch (inp.dtype()) {
            case DType::Float32:
                kernels::launch_layernorm_forward_kernel<float>(
                    static_cast<const float*>(inp.data()),
                    static_cast<float*>(out.data()),
                    static_cast<const float*>(w.data()),
                    static_cast<const float*>(b.data()),
                    batch_size, normalized_size, eps, inp.stream());
                break;
            case DType::Float64:
                kernels::launch_layernorm_forward_kernel<double>(
                    static_cast<const double*>(inp.data()),
                    static_cast<double*>(out.data()),
                    static_cast<const double*>(w.data()),
                    static_cast<const double*>(b.data()),
                    batch_size, normalized_size, eps, inp.stream());
                break;
            default:
                throw std::runtime_error("layernorm_forward: unsupported dtype");
        }
        return out;
    }, py::arg("input"), py::arg("weight"), py::arg("bias"),
       py::arg("normalized_size"), py::arg("eps") = 1e-5f);

    // ---- Functional: Optimizer steps ----

    m.def("sgd_update", [](
        Tensor& param, const Tensor& grad, py::object velocity_obj,
        float lr, float momentum, float weight_decay
    ) {
        int64_t n = param.numel();
        void* vel_ptr = nullptr;
        Tensor vel_tensor;
        if (!velocity_obj.is_none()) {
            vel_tensor = velocity_obj.cast<Tensor>();
            vel_ptr = const_cast<void*>(vel_tensor.data());
        }
        switch (param.dtype()) {
            case DType::Float32:
                kernels::launch_sgd_update_kernel<float>(
                    static_cast<float*>(param.data()),
                    static_cast<const float*>(grad.data()),
                    static_cast<float*>(vel_ptr),
                    n, lr, momentum, weight_decay, param.stream());
                break;
            case DType::Float64:
                kernels::launch_sgd_update_kernel<double>(
                    static_cast<double*>(param.data()),
                    static_cast<const double*>(grad.data()),
                    static_cast<double*>(vel_ptr),
                    n, lr, momentum, weight_decay, param.stream());
                break;
            default:
                throw std::runtime_error("sgd_update: unsupported dtype");
        }
    }, py::arg("param"), py::arg("grad"), py::arg("velocity") = py::none(),
       py::arg("lr") = 0.01f, py::arg("momentum") = 0.0f, py::arg("weight_decay") = 0.0f);

    m.def("adam_update", [](
        Tensor& param, const Tensor& grad, Tensor& m, Tensor& v,
        float lr, float beta1, float beta2, float eps, float weight_decay, int step
    ) {
        int64_t n = param.numel();
        switch (param.dtype()) {
            case DType::Float32:
                kernels::launch_adam_update_kernel<float>(
                    static_cast<float*>(param.data()),
                    static_cast<const float*>(grad.data()),
                    static_cast<float*>(m.data()),
                    static_cast<float*>(v.data()),
                    n, lr, beta1, beta2, eps, weight_decay, step, param.stream());
                break;
            case DType::Float64:
                kernels::launch_adam_update_kernel<double>(
                    static_cast<double*>(param.data()),
                    static_cast<const double*>(grad.data()),
                    static_cast<double*>(m.data()),
                    static_cast<double*>(v.data()),
                    n, lr, beta1, beta2, eps, weight_decay, step, param.stream());
                break;
            default:
                throw std::runtime_error("adam_update: unsupported dtype");
        }
    }, py::arg("param"), py::arg("grad"), py::arg("m"), py::arg("v"),
       py::arg("lr") = 1e-3f, py::arg("beta1") = 0.9f, py::arg("beta2") = 0.999f,
       py::arg("eps") = 1e-8f, py::arg("weight_decay") = 0.0f, py::arg("step") = 1);

    m.def("adamw_update", [](
        Tensor& param, const Tensor& grad, Tensor& m, Tensor& v,
        float lr, float beta1, float beta2, float eps, float weight_decay, int step
    ) {
        int64_t n = param.numel();
        switch (param.dtype()) {
            case DType::Float32:
                kernels::launch_adamw_update_kernel<float>(
                    static_cast<float*>(param.data()),
                    static_cast<const float*>(grad.data()),
                    static_cast<float*>(m.data()),
                    static_cast<float*>(v.data()),
                    n, lr, beta1, beta2, eps, weight_decay, step, param.stream());
                break;
            case DType::Float64:
                kernels::launch_adamw_update_kernel<double>(
                    static_cast<double*>(param.data()),
                    static_cast<const double*>(grad.data()),
                    static_cast<double*>(m.data()),
                    static_cast<double*>(v.data()),
                    n, lr, beta1, beta2, eps, weight_decay, step, param.stream());
                break;
            default:
                throw std::runtime_error("adamw_update: unsupported dtype");
        }
    }, py::arg("param"), py::arg("grad"), py::arg("m"), py::arg("v"),
       py::arg("lr") = 1e-3f, py::arg("beta1") = 0.9f, py::arg("beta2") = 0.999f,
       py::arg("eps") = 1e-8f, py::arg("weight_decay") = 1e-2f, py::arg("step") = 1);

    // ---- Device info ----

    m.def("device_info", []() {
        int device;
        cudaGetDevice(&device);
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, device);
        py::dict info;
        info["name"] = std::string(props.name);
        info["compute_capability"] = std::to_string(props.major) + "." + std::to_string(props.minor);
        info["total_memory_mb"] = static_cast<int>(props.totalGlobalMem / (1024 * 1024));
        info["multiprocessors"] = props.multiProcessorCount;
        info["max_threads_per_block"] = props.maxThreadsPerBlock;
        info["warp_size"] = props.warpSize;
        info["clock_rate_mhz"] = props.clockRate / 1000;
        info["memory_clock_rate_mhz"] = props.memoryClockRate / 1000;
        info["memory_bus_width"] = props.memoryBusWidth;
        return info;
    });

    // ---- Scalar mul (for dropout scaling etc.) ----

    // ---- Broadcast add: [N,M] + [M] -> [N,M] ----

    m.def("broadcast_add", [](const Tensor& matrix, const Tensor& row) {
        Tensor mat = ensure_contiguous(matrix);
        Tensor r = ensure_contiguous(row);
        auto& ms = mat.shape();
        auto& rs = r.shape();
        if (ms.size() != 2 || rs.size() != 1 || ms[1] != rs[0]) {
            throw std::runtime_error(
                "broadcast_add: requires matrix [N,M] and row [M], got incompatible shapes");
        }
        Tensor out(ms, mat.dtype());
        int64_t rows = ms[0], cols = ms[1];
        switch (mat.dtype()) {
            case DType::Float32:
                kernels::launch_broadcast_add_row_kernel<float>(
                    static_cast<const float*>(mat.data()),
                    static_cast<const float*>(r.data()),
                    static_cast<float*>(out.data()),
                    rows, cols, mat.stream());
                break;
            case DType::Float64:
                kernels::launch_broadcast_add_row_kernel<double>(
                    static_cast<const double*>(mat.data()),
                    static_cast<const double*>(r.data()),
                    static_cast<double*>(out.data()),
                    rows, cols, mat.stream());
                break;
            default:
                throw std::runtime_error("broadcast_add: unsupported dtype");
        }
        return out;
    }, py::arg("matrix"), py::arg("row"));

    m.def("scalar_mul", [](const Tensor& t, float scalar) {
        Tensor input = ensure_contiguous(t);
        Tensor out(input.shape(), input.dtype());
        int64_t n = input.numel();
        switch (input.dtype()) {
            case DType::Float32:
                kernels::launch_scalar_mul_kernel<float>(
                    static_cast<const float*>(input.data()), scalar,
                    static_cast<float*>(out.data()), n, input.stream());
                break;
            case DType::Float64:
                kernels::launch_scalar_mul_kernel<double>(
                    static_cast<const double*>(input.data()), scalar,
                    static_cast<double*>(out.data()), n, input.stream());
                break;
            default:
                throw std::runtime_error("scalar_mul: unsupported dtype");
        }
        return out;
    }, py::arg("input"), py::arg("scalar"));

    // ---- Unary ops (for autograd) ----

#define BIND_UNARY_OP(name, launch_fn)                                        \
    m.def(#name, [](const Tensor& t) {                                        \
        Tensor input = ensure_contiguous(t);                                   \
        Tensor out(input.shape(), input.dtype());                              \
        int64_t n = input.numel();                                             \
        switch (input.dtype()) {                                               \
            case DType::Float32:                                               \
                kernels::launch_fn<float>(                                     \
                    static_cast<const float*>(input.data()),                    \
                    static_cast<float*>(out.data()), n, input.stream());        \
                break;                                                         \
            case DType::Float64:                                               \
                kernels::launch_fn<double>(                                    \
                    static_cast<const double*>(input.data()),                   \
                    static_cast<double*>(out.data()), n, input.stream());       \
                break;                                                         \
            default:                                                           \
                throw std::runtime_error(#name ": unsupported dtype");          \
        }                                                                      \
        return out;                                                            \
    }, py::arg("input"));

    BIND_UNARY_OP(neg, launch_neg_kernel)
    BIND_UNARY_OP(exp_op, launch_exp_kernel)
    BIND_UNARY_OP(log_op, launch_log_kernel)
    BIND_UNARY_OP(sqrt_op, launch_sqrt_kernel)
    BIND_UNARY_OP(abs_op, launch_abs_kernel)
#undef BIND_UNARY_OP

    m.def("pow_op", [](const Tensor& t, float exponent) {
        Tensor input = ensure_contiguous(t);
        Tensor out(input.shape(), input.dtype());
        int64_t n = input.numel();
        switch (input.dtype()) {
            case DType::Float32:
                kernels::launch_pow_kernel<float>(
                    static_cast<const float*>(input.data()), exponent,
                    static_cast<float*>(out.data()), n, input.stream());
                break;
            case DType::Float64:
                kernels::launch_pow_kernel<double>(
                    static_cast<const double*>(input.data()), exponent,
                    static_cast<double*>(out.data()), n, input.stream());
                break;
            default:
                throw std::runtime_error("pow_op: unsupported dtype");
        }
        return out;
    }, py::arg("input"), py::arg("exponent"));

    m.def("clamp_op", [](const Tensor& t, float lo, float hi) {
        Tensor input = ensure_contiguous(t);
        Tensor out(input.shape(), input.dtype());
        int64_t n = input.numel();
        switch (input.dtype()) {
            case DType::Float32:
                kernels::launch_clamp_kernel<float>(
                    static_cast<const float*>(input.data()),
                    static_cast<float*>(out.data()), lo, hi, n, input.stream());
                break;
            case DType::Float64:
                kernels::launch_clamp_kernel<double>(
                    static_cast<const double*>(input.data()),
                    static_cast<double*>(out.data()), lo, hi, n, input.stream());
                break;
            default:
                throw std::runtime_error("clamp_op: unsupported dtype");
        }
        return out;
    }, py::arg("input"), py::arg("lo"), py::arg("hi"));

    m.def("gt_mask", [](const Tensor& t, float threshold) {
        Tensor input = ensure_contiguous(t);
        Tensor out(input.shape(), input.dtype());
        int64_t n = input.numel();
        switch (input.dtype()) {
            case DType::Float32:
                kernels::launch_gt_mask_kernel<float>(
                    static_cast<const float*>(input.data()), threshold,
                    static_cast<float*>(out.data()), n, input.stream());
                break;
            case DType::Float64:
                kernels::launch_gt_mask_kernel<double>(
                    static_cast<const double*>(input.data()), threshold,
                    static_cast<double*>(out.data()), n, input.stream());
                break;
            default:
                throw std::runtime_error("gt_mask: unsupported dtype");
        }
        return out;
    }, py::arg("input"), py::arg("threshold") = 0.0f);

    // Elementwise div (already have kernel, just need binding)
    m.def("div_op", [](const Tensor& a, const Tensor& b) {
        Tensor ia = ensure_contiguous(a);
        Tensor ib = ensure_contiguous(b);
        Tensor out(ia.shape(), ia.dtype());
        int64_t n = ia.numel();
        switch (ia.dtype()) {
            case DType::Float32:
                kernels::launch_div_kernel<float>(
                    static_cast<const float*>(ia.data()),
                    static_cast<const float*>(ib.data()),
                    static_cast<float*>(out.data()), n, ia.stream());
                break;
            case DType::Float64:
                kernels::launch_div_kernel<double>(
                    static_cast<const double*>(ia.data()),
                    static_cast<const double*>(ib.data()),
                    static_cast<double*>(out.data()), n, ia.stream());
                break;
            default:
                throw std::runtime_error("div_op: unsupported dtype");
        }
        return out;
    }, py::arg("a"), py::arg("b"));

    // ---- Activation backward (for autograd) ----
    m.def("activation_backward", [](const Tensor& grad_output, const Tensor& input,
                                     const std::string& act_name, float alpha) {
        Tensor go = ensure_contiguous(grad_output);
        Tensor inp = ensure_contiguous(input);
        Tensor grad_input(inp.shape(), inp.dtype());
        int64_t n = inp.numel();

        kernels::ActivationType act;
        if (act_name == "relu") act = kernels::ActivationType::ReLU;
        else if (act_name == "sigmoid") act = kernels::ActivationType::Sigmoid;
        else if (act_name == "tanh") act = kernels::ActivationType::Tanh;
        else if (act_name == "gelu") act = kernels::ActivationType::GELU;
        else if (act_name == "silu") act = kernels::ActivationType::SiLU;
        else if (act_name == "leaky_relu") act = kernels::ActivationType::LeakyReLU;
        else throw std::runtime_error("Unknown activation: " + act_name);

        switch (inp.dtype()) {
            case DType::Float32:
                kernels::launch_activation_backward_kernel<float>(
                    static_cast<const float*>(go.data()),
                    static_cast<const float*>(inp.data()),
                    static_cast<float*>(grad_input.data()),
                    n, act, alpha, inp.stream());
                break;
            case DType::Float64:
                kernels::launch_activation_backward_kernel<double>(
                    static_cast<const double*>(go.data()),
                    static_cast<const double*>(inp.data()),
                    static_cast<double*>(grad_input.data()),
                    n, act, alpha, inp.stream());
                break;
            default:
                throw std::runtime_error("activation_backward: unsupported dtype");
        }
        return grad_input;
    }, py::arg("grad_output"), py::arg("input"), py::arg("act_name"), py::arg("alpha") = 0.01f);

    // ---- Row-sum reduction: [N,M] -> [M] (sum along axis 0) ----
    m.def("sum_reduce_rows", [](const Tensor& t) {
        Tensor input = ensure_contiguous(t);
        auto& shape = input.shape();
        int64_t rows = shape[0], cols = shape[1];
        Tensor out({cols}, input.dtype());
        switch (input.dtype()) {
            case DType::Float32:
                kernels::launch_sum_reduce_rows_kernel<float>(
                    static_cast<const float*>(input.data()),
                    static_cast<float*>(out.data()), rows, cols, input.stream());
                break;
            case DType::Float64:
                kernels::launch_sum_reduce_rows_kernel<double>(
                    static_cast<const double*>(input.data()),
                    static_cast<double*>(out.data()), rows, cols, input.stream());
                break;
            default:
                throw std::runtime_error("sum_reduce_rows: unsupported dtype");
        }
        return out;
    }, py::arg("input"));

    // ---- Conv2d backward (for autograd) ----
    m.def("conv2d_backward_data", [](
        const Tensor& grad_output, const Tensor& weight,
        std::vector<int64_t> input_shape,
        std::vector<int> stride, std::vector<int> padding
    ) {
        Tensor go = ensure_contiguous(grad_output);
        Tensor w = ensure_contiguous(weight);
        auto& wshape = w.shape();
        int B  = static_cast<int>(input_shape[0]);
        int IC = static_cast<int>(input_shape[1]);
        int IH = static_cast<int>(input_shape[2]);
        int IW = static_cast<int>(input_shape[3]);
        int OC = static_cast<int>(wshape[0]);
        int KH = static_cast<int>(wshape[2]);
        int KW = static_cast<int>(wshape[3]);
        int SH = stride[0], SW = stride[1];
        int PH = padding[0], PW = padding[1];

        Tensor grad_input(input_shape, go.dtype());
        grad_input.zero_();
        switch (go.dtype()) {
            case DType::Float32:
                kernels::launch_conv2d_backward_data_kernel<float>(
                    static_cast<const float*>(go.data()),
                    static_cast<const float*>(w.data()),
                    static_cast<float*>(grad_input.data()),
                    B, IC, OC, IH, IW, KH, KW, SH, SW, PH, PW, go.stream());
                break;
            case DType::Float64:
                kernels::launch_conv2d_backward_data_kernel<double>(
                    static_cast<const double*>(go.data()),
                    static_cast<const double*>(w.data()),
                    static_cast<double*>(grad_input.data()),
                    B, IC, OC, IH, IW, KH, KW, SH, SW, PH, PW, go.stream());
                break;
            default:
                throw std::runtime_error("conv2d_backward_data: unsupported dtype");
        }
        return grad_input;
    }, py::arg("grad_output"), py::arg("weight"),
       py::arg("input_shape"), py::arg("stride"), py::arg("padding"));

    m.def("conv2d_backward_weight", [](
        const Tensor& grad_output, const Tensor& input,
        std::vector<int64_t> weight_shape,
        std::vector<int> stride, std::vector<int> padding
    ) {
        Tensor go = ensure_contiguous(grad_output);
        Tensor inp = ensure_contiguous(input);
        auto& ishape = inp.shape();
        int B  = static_cast<int>(ishape[0]);
        int IC = static_cast<int>(ishape[1]);
        int IH = static_cast<int>(ishape[2]);
        int IW = static_cast<int>(ishape[3]);
        int OC = static_cast<int>(weight_shape[0]);
        int KH = static_cast<int>(weight_shape[2]);
        int KW = static_cast<int>(weight_shape[3]);
        int SH = stride[0], SW = stride[1];
        int PH = padding[0], PW = padding[1];

        Tensor grad_weight(weight_shape, go.dtype());
        grad_weight.zero_();
        switch (go.dtype()) {
            case DType::Float32:
                kernels::launch_conv2d_backward_weight_kernel<float>(
                    static_cast<const float*>(go.data()),
                    static_cast<const float*>(inp.data()),
                    static_cast<float*>(grad_weight.data()),
                    B, IC, OC, IH, IW, KH, KW, SH, SW, PH, PW, go.stream());
                break;
            case DType::Float64:
                kernels::launch_conv2d_backward_weight_kernel<double>(
                    static_cast<const double*>(go.data()),
                    static_cast<const double*>(inp.data()),
                    static_cast<double*>(grad_weight.data()),
                    B, IC, OC, IH, IW, KH, KW, SH, SW, PH, PW, go.stream());
                break;
            default:
                throw std::runtime_error("conv2d_backward_weight: unsupported dtype");
        }
        return grad_weight;
    }, py::arg("grad_output"), py::arg("input"),
       py::arg("weight_shape"), py::arg("stride"), py::arg("padding"));
}
