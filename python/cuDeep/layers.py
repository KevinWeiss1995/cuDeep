"""Additional layers: normalization, pooling, dropout, embedding."""

from __future__ import annotations
from cuDeep._core import (
    DType,
    max_pool2d as _max_pool2d,
    avg_pool2d as _avg_pool2d,
    batchnorm_forward as _batchnorm_forward,
    layernorm_forward as _layernorm_forward,
    scalar_mul as _scalar_mul,
    Tensor as _RawTensor,
)
from cuDeep.tensor import Tensor
from cuDeep.nn import Module, Parameter


class BatchNorm2d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, dtype=DType.float32):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.register_parameter("weight", Tensor.ones([num_features], dtype))
        self.register_parameter("bias", Tensor.zeros([num_features], dtype))
        self.running_mean = _RawTensor.zeros([num_features], dtype)
        self.running_var = _RawTensor.ones([num_features], dtype)

    def forward(self, x):
        out = _batchnorm_forward(
            x._data,
            self._parameters["weight"]._data,
            self._parameters["bias"]._data,
            self.running_mean,
            self.running_var,
            self.eps, self.momentum, self._training)
        return Tensor._wrap(out)


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5, dtype=DType.float32):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = [normalized_shape]
        self.normalized_shape = normalized_shape
        self.eps = eps
        size = 1
        for s in normalized_shape:
            size *= s
        self._normalized_size = size
        self.register_parameter("weight", Tensor.ones([size], dtype))
        self.register_parameter("bias", Tensor.zeros([size], dtype))

    def forward(self, x):
        out = _layernorm_forward(
            x._data,
            self._parameters["weight"]._data,
            self._parameters["bias"]._data,
            self._normalized_size, self.eps)
        return Tensor._wrap(out)


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride or self.kernel_size
        if isinstance(self.stride, int):
            self.stride = (self.stride, self.stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)

    def forward(self, x):
        out = _max_pool2d(x._data,
                          list(self.kernel_size), list(self.stride), list(self.padding))
        return Tensor._wrap(out)


class AvgPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride or self.kernel_size
        if isinstance(self.stride, int):
            self.stride = (self.stride, self.stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)

    def forward(self, x):
        out = _avg_pool2d(x._data,
                          list(self.kernel_size), list(self.stride), list(self.padding))
        return Tensor._wrap(out)


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self._training or self.p == 0.0:
            return x
        return Tensor._wrap(_scalar_mul(x._data, 1.0 / (1.0 - self.p)))


class Flatten(Module):
    """Flatten all dims after the batch dim."""
    def __init__(self, start_dim=1):
        super().__init__()
        self.start_dim = start_dim

    def forward(self, x):
        shape = x.shape()
        flat_size = 1
        for d in shape[self.start_dim:]:
            flat_size *= d
        return x.reshape(list(shape[:self.start_dim]) + [flat_size])


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, dtype=DType.float32):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.register_parameter(
            "weight", Tensor.randn([num_embeddings, embedding_dim], dtype))

    def forward(self, indices):
        raise NotImplementedError("Embedding lookup kernel planned for v0.2")
