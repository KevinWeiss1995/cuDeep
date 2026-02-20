"""
Additional layer definitions beyond core nn.py.

Houses normalization, pooling, dropout, and embedding layers.
"""

from __future__ import annotations

import random

from cuDeep._core import (
    Tensor, DType,
    max_pool2d, avg_pool2d,
    batchnorm_forward, layernorm_forward,
    scalar_mul,
)
from cuDeep.nn import Module


class BatchNorm2d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, dtype=DType.float32):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.register_parameter("weight", Tensor.ones([num_features], dtype))
        self.register_parameter("bias", Tensor.zeros([num_features], dtype))
        self.running_mean = Tensor.zeros([num_features], dtype)
        self.running_var = Tensor.ones([num_features], dtype)

    def forward(self, x):
        return batchnorm_forward(
            x,
            self._parameters["weight"],
            self._parameters["bias"],
            self.running_mean,
            self.running_var,
            self.eps,
            self.momentum,
            self._training)


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
        return layernorm_forward(
            x,
            self._parameters["weight"],
            self._parameters["bias"],
            self._normalized_size,
            self.eps)


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride or self.kernel_size
        if isinstance(self.stride, int):
            self.stride = (self.stride, self.stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)

    def forward(self, x):
        return max_pool2d(
            x,
            list(self.kernel_size),
            list(self.stride),
            list(self.padding))


class AvgPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride or self.kernel_size
        if isinstance(self.stride, int):
            self.stride = (self.stride, self.stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)

    def forward(self, x):
        return avg_pool2d(
            x,
            list(self.kernel_size),
            list(self.stride),
            list(self.padding))


class Dropout(Module):
    """Inverted dropout using scalar_mul with random masking.

    Note: True per-element dropout requires a CUDA random mask kernel.
    This implementation scales the entire tensor — functional for inference
    (pass-through) and as a training placeholder. A proper per-element CUDA
    dropout kernel is planned for v0.2.
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self._training or self.p == 0.0:
            return x
        return scalar_mul(x, 1.0 / (1.0 - self.p))


class Embedding(Module):
    """Embedding lookup table.

    Note: Full GPU embedding lookup requires a dedicated kernel (planned v0.2).
    Weights are stored on GPU and can be used as parameters.
    """

    def __init__(self, num_embeddings, embedding_dim, dtype=DType.float32):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.register_parameter(
            "weight", Tensor.randn([num_embeddings, embedding_dim], dtype)
        )

    def forward(self, indices):
        raise NotImplementedError(
            "Embedding lookup requires dedicated CUDA kernel (planned v0.2)")
