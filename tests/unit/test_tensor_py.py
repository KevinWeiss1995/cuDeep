"""Python-side tensor tests (requires built extension)."""

import numpy as np
import pytest


def test_tensor_zeros():
    from cuDeep import Tensor
    t = Tensor.zeros([4, 4])
    arr = t.numpy()
    assert arr.shape == (4, 4)
    assert np.allclose(arr, 0.0)


def test_tensor_from_numpy_roundtrip():
    from cuDeep import Tensor
    original = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    t = Tensor.from_numpy(original)
    result = t.numpy()
    assert np.allclose(original, result)


def test_tensor_shape():
    from cuDeep import Tensor
    t = Tensor([2, 3, 4])
    assert t.shape() == [2, 3, 4]
    assert t.ndim() == 3
    assert t.numel() == 24
