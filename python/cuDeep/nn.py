"""Neural network layers and modules."""

from __future__ import annotations

from typing import Optional

from cuDeep._core import (
    Tensor, DType,
    relu, sigmoid, gelu, silu, tanh_act, leaky_relu,
    conv2d_forward, broadcast_add,
)


class Module:
    """Base class for all neural network modules."""

    def __init__(self):
        self._parameters = {}
        self._modules = {}
        self._training = True

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def parameters(self):
        params = list(self._parameters.values())
        for module in self._modules.values():
            params.extend(module.parameters())
        return params

    def train(self, mode=True):
        self._training = mode
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def register_parameter(self, name, param):
        self._parameters[name] = param

    def register_module(self, name, module):
        self._modules[name] = module


class Linear(Module):
    """Fully connected layer: y = xW^T + b."""

    def __init__(self, in_features, out_features, bias=True, dtype=DType.float32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_parameter("weight", Tensor.randn([out_features, in_features], dtype))
        if bias:
            self.register_parameter("bias", Tensor.zeros([out_features], dtype))

    def forward(self, x):
        out = x.matmul(self._parameters["weight"].transpose(0, 1).contiguous())
        if "bias" in self._parameters:
            out = broadcast_add(out, self._parameters["bias"])
        return out


class Conv2d(Module):
    """2D convolution layer."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True, dtype=DType.float32):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.register_parameter(
            "weight",
            Tensor.randn([out_channels, in_channels, kernel_size[0], kernel_size[1]], dtype),
        )
        if bias:
            self.register_parameter("bias", Tensor.zeros([out_channels], dtype))

    def forward(self, x):
        bias = self._parameters.get("bias", None)
        return conv2d_forward(
            x, self._parameters["weight"], bias,
            list(self.stride), list(self.padding))


class ReLU(Module):
    def forward(self, x):
        return relu(x)


class Sigmoid(Module):
    def forward(self, x):
        return sigmoid(x)


class Tanh(Module):
    def forward(self, x):
        return tanh_act(x)


class GELU(Module):
    def forward(self, x):
        return gelu(x)


class SiLU(Module):
    def forward(self, x):
        return silu(x)


class LeakyReLU(Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        return leaky_relu(x, self.negative_slope)


class Sequential(Module):
    """Sequential container that chains modules in order."""

    def __init__(self, *modules):
        super().__init__()
        for i, mod in enumerate(modules):
            self.register_module(str(i), mod)

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x
