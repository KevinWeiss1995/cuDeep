"""
Additional layer definitions beyond core nn.py.

Houses normalization, pooling, dropout, embedding, and attention layers.
"""

from __future__ import annotations

from cuDeep._core import Tensor, DType
from cuDeep.nn import Module


class BatchNorm2d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, dtype=DType.float32):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.register_parameter("weight", Tensor.ones([num_features], dtype))
        self.register_parameter("bias", Tensor.zeros([num_features], dtype))

    def forward(self, x):
        raise NotImplementedError


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5, dtype=DType.float32):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        size = 1
        for s in normalized_shape:
            size *= s
        self.register_parameter("weight", Tensor.ones([size], dtype))
        self.register_parameter("bias", Tensor.zeros([size], dtype))

    def forward(self, x):
        raise NotImplementedError


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride or self.kernel_size
        if isinstance(self.stride, int):
            self.stride = (self.stride, self.stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)

    def forward(self, x):
        raise NotImplementedError


class AvgPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride or self.kernel_size
        if isinstance(self.stride, int):
            self.stride = (self.stride, self.stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)

    def forward(self, x):
        raise NotImplementedError


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self._training or self.p == 0.0:
            return x
        raise NotImplementedError


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, dtype=DType.float32):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.register_parameter(
            "weight", Tensor.randn([num_embeddings, embedding_dim], dtype)
        )

    def forward(self, indices):
        raise NotImplementedError
