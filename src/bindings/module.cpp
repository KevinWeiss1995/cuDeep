#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "cudeep/tensor.cuh"
#include "cudeep/memory.cuh"
#include "cudeep/stream.cuh"
#include "cudeep/common.cuh"

namespace py = pybind11;

using namespace cudeep;

PYBIND11_MODULE(_cudeep_core, m) {
    m.doc() = "cuDeep: Ultra-high performance deep learning library";

    py::enum_<DType>(m, "DType")
        .value("float16", DType::Float16)
        .value("float32", DType::Float32)
        .value("float64", DType::Float64);

    py::enum_<Layout>(m, "Layout")
        .value("NCHW", Layout::NCHW)
        .value("NHWC", Layout::NHWC);

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
        .def("numpy", [](const Tensor& t) {
            auto dtype = t.dtype();
            auto shape = t.shape();
            std::vector<ssize_t> py_shape(shape.begin(), shape.end());

            if (dtype == DType::Float32) {
                auto result = py::array_t<float>(py_shape);
                t.to_host(result.mutable_data());
                return py::cast(result);
            } else if (dtype == DType::Float64) {
                auto result = py::array_t<double>(py_shape);
                t.to_host(result.mutable_data());
                return py::cast(result);
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

    py::class_<Stream>(m, "Stream")
        .def(py::init<>())
        .def("synchronize", &Stream::synchronize);

    py::class_<Event>(m, "Event")
        .def(py::init<>())
        .def("record", &Event::record, py::arg("stream") = nullptr)
        .def("synchronize", &Event::synchronize)
        .def("elapsed_ms", &Event::elapsed_ms);

    py::class_<MemoryPool>(m, "MemoryPool")
        .def_static("instance", &MemoryPool::instance, py::return_value_policy::reference)
        .def("release_cached", &MemoryPool::release_cached)
        .def_property_readonly("allocated_bytes", &MemoryPool::allocated_bytes)
        .def_property_readonly("cached_bytes", &MemoryPool::cached_bytes);
}
