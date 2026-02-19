"""High-level tensor utilities wrapping the native Tensor class."""

from __future__ import annotations

import numpy as np
from cuDeep._core import Tensor as _Tensor
from cuDeep._core import DType


def from_numpy(arr):
    """Create a cuDeep Tensor from a NumPy array (copies data to GPU)."""
    arr = np.ascontiguousarray(arr)
    return _Tensor.from_numpy(arr)


def to_numpy(tensor):
    """Copy a cuDeep Tensor back to a NumPy array on the host."""
    return tensor.numpy()


def zeros(shape, dtype=DType.float32):
    return _Tensor.zeros(list(shape), dtype)


def ones(shape, dtype=DType.float32):
    return _Tensor.ones(list(shape), dtype)


def randn(shape, dtype=DType.float32):
    return _Tensor.randn(list(shape), dtype)
