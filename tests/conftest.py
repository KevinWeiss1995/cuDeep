"""Shared pytest fixtures for cuDeep tests."""

import numpy as np
import pytest


@pytest.fixture
def dtype_float32():
    from cuDeep import DType
    return DType.float32


@pytest.fixture
def dtype_float64():
    from cuDeep import DType
    return DType.float64


@pytest.fixture
def random_matrix():
    """Returns a factory that creates matching numpy/cuDeep tensor pairs."""
    from cuDeep import Tensor

    def _make(rows, cols, dtype=np.float32):
        arr = np.random.randn(rows, cols).astype(dtype)
        t = Tensor.from_numpy(arr)
        return arr, t

    return _make
