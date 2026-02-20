"""Weight initialization utilities (Xavier, Kaiming, etc.)."""

from __future__ import annotations

import math
import numpy as np

from cuDeep.tensor import Tensor
from cuDeep._core import DType, Tensor as _RawTensor


def _fan_in_out(tensor: Tensor):
    shape = tensor.shape()
    ndim = len(shape)
    if ndim < 2:
        raise ValueError("fan_in/fan_out requires at least 2D tensor")
    fan_in = shape[1]
    fan_out = shape[0]
    if ndim > 2:
        receptive = 1
        for s in shape[2:]:
            receptive *= s
        fan_in *= receptive
        fan_out *= receptive
    return fan_in, fan_out


def xavier_uniform_(tensor: Tensor, gain=1.0):
    fan_in, fan_out = _fan_in_out(tensor)
    bound = gain * math.sqrt(6.0 / (fan_in + fan_out))
    arr = np.random.uniform(-bound, bound, tensor.shape()).astype(np.float32)
    tensor._data = _RawTensor.from_numpy(arr)
    return tensor


def xavier_normal_(tensor: Tensor, gain=1.0):
    fan_in, fan_out = _fan_in_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    arr = np.random.normal(0, std, tensor.shape()).astype(np.float32)
    tensor._data = _RawTensor.from_numpy(arr)
    return tensor


def kaiming_uniform_(tensor: Tensor, a=0, mode="fan_in", nonlinearity="leaky_relu"):
    fan_in, fan_out = _fan_in_out(tensor)
    fan = fan_in if mode == "fan_in" else fan_out
    gain = math.sqrt(2.0 / (1 + a ** 2))
    bound = gain * math.sqrt(3.0 / fan)
    arr = np.random.uniform(-bound, bound, tensor.shape()).astype(np.float32)
    tensor._data = _RawTensor.from_numpy(arr)
    return tensor


def kaiming_normal_(tensor: Tensor, a=0, mode="fan_in", nonlinearity="leaky_relu"):
    fan_in, fan_out = _fan_in_out(tensor)
    fan = fan_in if mode == "fan_in" else fan_out
    gain = math.sqrt(2.0 / (1 + a ** 2))
    std = gain / math.sqrt(fan)
    arr = np.random.normal(0, std, tensor.shape()).astype(np.float32)
    tensor._data = _RawTensor.from_numpy(arr)
    return tensor


def zeros_(tensor: Tensor):
    tensor._data.zero_()
    return tensor


def ones_(tensor: Tensor):
    tensor._data.fill_(1.0)
    return tensor


def constant_(tensor: Tensor, val):
    tensor._data.fill_(float(val))
    return tensor


def uniform_(tensor: Tensor, a=0.0, b=1.0):
    arr = np.random.uniform(a, b, tensor.shape()).astype(np.float32)
    tensor._data = _RawTensor.from_numpy(arr)
    return tensor


def normal_(tensor: Tensor, mean=0.0, std=1.0):
    arr = np.random.normal(mean, std, tensor.shape()).astype(np.float32)
    tensor._data = _RawTensor.from_numpy(arr)
    return tensor
