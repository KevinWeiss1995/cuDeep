"""Neural network modules with autograd support.

All modules operate on ``cuDeep.tensor.Tensor`` objects. Parameters are
created with ``requires_grad=True`` so that ``loss.backward()`` populates
their ``.grad`` attributes automatically.
"""

from __future__ import annotations

from cuDeep._core import DType
from cuDeep.tensor import (
    Tensor, broadcast_add, relu, sigmoid, tanh, gelu, silu, softmax,
    conv2d, Conv2dOp,
)


class Parameter(Tensor):
    """A tensor that is automatically marked as requiring gradients."""

    def __init__(self, data):
        if isinstance(data, Tensor):
            super().__init__(data._data, requires_grad=True)
        else:
            super().__init__(data, requires_grad=True)

    def __repr__(self):
        return f"Parameter(shape={self.shape()}, dtype={self.dtype()})"


class Module:
    """Base class for all neural network modules."""

    def __init__(self):
        self._parameters: dict[str, Parameter] = {}
        self._modules: dict[str, Module] = {}
        self._training = True

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def register_parameter(self, name: str, tensor):
        if not isinstance(tensor, Parameter):
            tensor = Parameter(tensor)
        self._parameters[name] = tensor

    def register_module(self, name: str, module: Module):
        self._modules[name] = module

    def parameters(self) -> list[Parameter]:
        params = list(self._parameters.values())
        for mod in self._modules.values():
            params.extend(mod.parameters())
        return params

    def named_parameters(self):
        for name, p in self._parameters.items():
            yield name, p
        for mod_name, mod in self._modules.items():
            for pname, p in mod.named_parameters():
                yield f"{mod_name}.{pname}", p

    def train(self, mode=True):
        self._training = mode
        for m in self._modules.values():
            m.train(mode)
        return self

    def eval(self):
        return self.train(False)

    @property
    def training(self):
        return self._training

    def state_dict(self) -> dict:
        sd = {}
        for name, p in self.named_parameters():
            sd[name] = p.numpy()
        return sd

    def load_state_dict(self, sd: dict):
        import numpy as np
        params = dict(self.named_parameters())
        for name, arr in sd.items():
            if name in params:
                from cuDeep._core import Tensor as _RT
                params[name]._data = _RT.from_numpy(np.ascontiguousarray(arr))

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None


class Linear(Module):
    """Fully connected layer: y = xW^T + b."""

    def __init__(self, in_features, out_features, bias=True, dtype=DType.float32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_parameter(
            "weight", Tensor.randn([out_features, in_features], dtype))
        if bias:
            self.register_parameter(
                "bias", Tensor.zeros([out_features], dtype))

    def forward(self, x):
        w = self._parameters["weight"]
        out = x.matmul(w.transpose(0, 1))
        if "bias" in self._parameters:
            out = broadcast_add(out, self._parameters["bias"])
        return out


class Conv2d(Module):
    """2D convolution."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True, dtype=DType.float32):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        self.stride = stride
        self.padding = padding
        self.register_parameter(
            "weight",
            Tensor.randn([out_channels, in_channels, kernel_size[0], kernel_size[1]], dtype))
        if bias:
            self.register_parameter(
                "bias", Tensor.zeros([out_channels], dtype))

    def forward(self, x):
        b = self._parameters.get("bias", None)
        return conv2d(x, self._parameters["weight"], b,
                      self.stride, self.padding)


class ReLU(Module):
    def forward(self, x): return x.relu()

class Sigmoid(Module):
    def forward(self, x): return x.sigmoid()

class Tanh(Module):
    def forward(self, x): return x.tanh()

class GELU(Module):
    def forward(self, x): return x.gelu()

class SiLU(Module):
    def forward(self, x): return x.silu()

class LeakyReLU(Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope
    def forward(self, x):
        from cuDeep.tensor import _LeakyReLUAlphaOp
        return _LeakyReLUAlphaOp.apply(x, self.negative_slope)

class Softmax(Module):
    def forward(self, x): return softmax(x)


class Sequential(Module):
    """Run modules in sequence."""

    def __init__(self, *modules):
        super().__init__()
        for i, m in enumerate(modules):
            self.register_module(str(i), m)

    def forward(self, x):
        for m in self._modules.values():
            x = m(x)
        return x

    def __getitem__(self, idx):
        return list(self._modules.values())[idx]

    def __len__(self):
        return len(self._modules)
