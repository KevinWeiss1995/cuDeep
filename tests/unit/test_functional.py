"""Tests for the functional API — activations, reductions, losses, softmax."""

import numpy as np
import pytest

from cuDeep import DType
from cuDeep._core import (
    Tensor as _RT,
    relu, sigmoid, tanh_act, gelu, silu, leaky_relu,
    sum as cu_sum, mean as cu_mean, max as cu_max, min as cu_min,
    softmax, mse_loss, cross_entropy_loss,
    broadcast_add, scalar_mul,
)


# ---- Activations ----

class TestReLU:
    def test_positive_passthrough(self):
        data = np.array([1, 2, 3], dtype=np.float32)
        t = _RT.from_numpy(data)
        out = relu(t).numpy()
        np.testing.assert_allclose(out, data)

    def test_negative_zeroed(self):
        data = np.array([-1, -2, -3], dtype=np.float32)
        t = _RT.from_numpy(data)
        out = relu(t).numpy()
        np.testing.assert_array_equal(out, 0.0)

    def test_mixed(self):
        data = np.array([-3, -1, 0, 1, 3], dtype=np.float32)
        t = _RT.from_numpy(data)
        out = relu(t).numpy()
        np.testing.assert_allclose(out, np.maximum(data, 0))

    def test_large(self):
        data = np.random.randn(1024).astype(np.float32)
        t = _RT.from_numpy(data)
        out = relu(t).numpy()
        np.testing.assert_allclose(out, np.maximum(data, 0))


class TestSigmoid:
    def test_zero(self):
        t = _RT.from_numpy(np.array([0.0], dtype=np.float32))
        out = sigmoid(t).numpy()
        np.testing.assert_allclose(out, [0.5], atol=1e-5)

    def test_large_positive(self):
        t = _RT.from_numpy(np.array([100.0], dtype=np.float32))
        out = sigmoid(t).numpy()
        assert out[0] > 0.999

    def test_large_negative(self):
        t = _RT.from_numpy(np.array([-100.0], dtype=np.float32))
        out = sigmoid(t).numpy()
        assert out[0] < 0.001

    def test_range(self):
        data = np.linspace(-5, 5, 100).astype(np.float32)
        t = _RT.from_numpy(data)
        out = sigmoid(t).numpy()
        expected = 1.0 / (1.0 + np.exp(-data))
        np.testing.assert_allclose(out, expected, rtol=1e-5)


class TestGELU:
    def test_zero(self):
        t = _RT.from_numpy(np.array([0.0], dtype=np.float32))
        out = gelu(t).numpy()
        np.testing.assert_allclose(out, [0.0], atol=1e-5)

    def test_positive(self):
        t = _RT.from_numpy(np.array([1.0], dtype=np.float32))
        out = gelu(t).numpy()
        assert 0.8 < out[0] < 0.85

    def test_symmetric_property(self):
        data = np.array([2.0, -2.0], dtype=np.float32)
        t = _RT.from_numpy(data)
        out = gelu(t).numpy()
        assert out[0] > 0
        assert out[1] < 0
        assert abs(out[0]) > abs(out[1])


class TestLeakyReLU:
    def test_positive(self):
        data = np.array([1.0, 2.0], dtype=np.float32)
        t = _RT.from_numpy(data)
        out = leaky_relu(t, 0.1).numpy()
        np.testing.assert_allclose(out, data)

    def test_negative(self):
        data = np.array([-10.0, -5.0], dtype=np.float32)
        t = _RT.from_numpy(data)
        out = leaky_relu(t, 0.1).numpy()
        np.testing.assert_allclose(out, data * 0.1, rtol=1e-5)


class TestSiLU:
    def test_zero(self):
        t = _RT.from_numpy(np.array([0.0], dtype=np.float32))
        out = silu(t).numpy()
        np.testing.assert_allclose(out, [0.0], atol=1e-5)

    def test_positive(self):
        data = np.array([1.0], dtype=np.float32)
        t = _RT.from_numpy(data)
        out = silu(t).numpy()
        expected = 1.0 / (1.0 + np.exp(-1.0))
        np.testing.assert_allclose(out, [expected], rtol=1e-5)


class TestTanh:
    def test_range(self):
        data = np.linspace(-3, 3, 50).astype(np.float32)
        t = _RT.from_numpy(data)
        out = tanh_act(t).numpy()
        np.testing.assert_allclose(out, np.tanh(data), rtol=1e-5)


# ---- Reductions ----

class TestSum:
    def test_simple(self):
        data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        t = _RT.from_numpy(data)
        result = cu_sum(t).numpy()
        np.testing.assert_allclose(result, [15.0], rtol=1e-5)

    def test_large(self):
        data = np.ones(4096, dtype=np.float32)
        t = _RT.from_numpy(data)
        result = cu_sum(t).numpy()
        np.testing.assert_allclose(result, [4096.0], rtol=1e-3)


class TestMean:
    def test_simple(self):
        data = np.array([2, 4, 6, 8], dtype=np.float32)
        t = _RT.from_numpy(data)
        result = cu_mean(t).numpy()
        np.testing.assert_allclose(result, [5.0], rtol=1e-5)

    def test_ones(self):
        t = _RT.ones([1000])
        result = cu_mean(t).numpy()
        np.testing.assert_allclose(result, [1.0], rtol=1e-3)


class TestMax:
    def test_simple(self):
        data = np.array([3, 1, 4, 1, 5, 9], dtype=np.float32)
        t = _RT.from_numpy(data)
        result = cu_max(t).numpy()
        np.testing.assert_allclose(result, [9.0])

    def test_negative(self):
        data = np.array([-5, -3, -1, -7], dtype=np.float32)
        t = _RT.from_numpy(data)
        result = cu_max(t).numpy()
        np.testing.assert_allclose(result, [-1.0])

    def test_large(self):
        data = np.random.randn(10000).astype(np.float32)
        t = _RT.from_numpy(data)
        result = cu_max(t).numpy()
        np.testing.assert_allclose(result, [data.max()], rtol=1e-4)


class TestMin:
    def test_simple(self):
        data = np.array([3, 1, 4, 1, 5], dtype=np.float32)
        t = _RT.from_numpy(data)
        result = cu_min(t).numpy()
        np.testing.assert_allclose(result, [1.0])

    def test_large(self):
        data = np.random.randn(10000).astype(np.float32)
        t = _RT.from_numpy(data)
        result = cu_min(t).numpy()
        np.testing.assert_allclose(result, [data.min()], rtol=1e-4)


# ---- Softmax ----

class TestSoftmax:
    def test_sums_to_one(self):
        data = np.array([[1, 2, 3]], dtype=np.float32)
        t = _RT.from_numpy(data)
        out = softmax(t, 1).numpy()
        np.testing.assert_allclose(out.sum(), 1.0, rtol=1e-5)

    def test_ordering_preserved(self):
        data = np.array([[1, 2, 3]], dtype=np.float32)
        t = _RT.from_numpy(data)
        out = softmax(t, 1).numpy()[0]
        assert out[0] < out[1] < out[2]

    def test_uniform(self):
        data = np.array([[1, 1, 1]], dtype=np.float32)
        t = _RT.from_numpy(data)
        out = softmax(t, 1).numpy()[0]
        np.testing.assert_allclose(out, [1/3, 1/3, 1/3], rtol=1e-5)

    def test_batch(self):
        data = np.array([[0, 0], [100, 0]], dtype=np.float32)
        t = _RT.from_numpy(data)
        out = softmax(t, 1).numpy()
        np.testing.assert_allclose(out[0], [0.5, 0.5], rtol=1e-5)
        assert out[1, 0] > 0.99
        assert out[1, 1] < 0.01


# ---- Loss functions ----

class TestMSELoss:
    def test_zero_loss(self):
        pred = _RT.from_numpy(np.array([1, 2, 3], dtype=np.float32))
        target = _RT.from_numpy(np.array([1, 2, 3], dtype=np.float32))
        loss = mse_loss(pred, target).numpy()
        np.testing.assert_allclose(loss, [0.0], atol=1e-5)

    def test_nonzero_loss(self):
        pred = _RT.from_numpy(np.array([1, 2, 3], dtype=np.float32))
        target = _RT.from_numpy(np.array([2, 3, 4], dtype=np.float32))
        loss = mse_loss(pred, target).numpy()
        np.testing.assert_allclose(loss, [1.0], rtol=1e-5)

    def test_known_value(self):
        pred = _RT.from_numpy(np.array([0, 0], dtype=np.float32))
        target = _RT.from_numpy(np.array([3, 4], dtype=np.float32))
        loss = mse_loss(pred, target).numpy()
        expected = (9.0 + 16.0) / 2.0
        np.testing.assert_allclose(loss, [expected], rtol=1e-4)


class TestCrossEntropyLoss:
    def test_perfect_prediction(self):
        logits = np.array([[100, 0, 0]], dtype=np.float32)
        targets = np.array([0], dtype=np.int32)
        t_logits = _RT.from_numpy(logits)
        loss = cross_entropy_loss(t_logits, targets).numpy()
        np.testing.assert_allclose(loss, [0.0], atol=0.01)

    def test_wrong_prediction(self):
        logits = np.array([[0, 0, 100]], dtype=np.float32)
        targets = np.array([0], dtype=np.int32)
        t_logits = _RT.from_numpy(logits)
        loss = cross_entropy_loss(t_logits, targets).numpy()
        assert loss[0] > 50.0

    def test_uniform_logits(self):
        logits = np.array([[0, 0, 0]], dtype=np.float32)
        targets = np.array([1], dtype=np.int32)
        t_logits = _RT.from_numpy(logits)
        loss = cross_entropy_loss(t_logits, targets).numpy()
        expected = np.log(3.0)
        np.testing.assert_allclose(loss, [expected], rtol=1e-4)

    def test_batch(self):
        logits = np.array([[10, 0], [0, 10]], dtype=np.float32)
        targets = np.array([0, 1], dtype=np.int32)
        t_logits = _RT.from_numpy(logits)
        loss = cross_entropy_loss(t_logits, targets).numpy()
        np.testing.assert_allclose(loss, [0.0], atol=0.01)


# ---- Scalar mul ----

class TestBroadcastAdd:
    def test_basic(self):
        matrix = np.array([[1, 2], [3, 4]], dtype=np.float32)
        row = np.array([10, 20], dtype=np.float32)
        tm = _RT.from_numpy(matrix)
        tr = _RT.from_numpy(row)
        out = broadcast_add(tm, tr).numpy()
        np.testing.assert_allclose(out, matrix + row)

    def test_large_batch(self):
        matrix = np.ones((100, 50), dtype=np.float32)
        row = np.arange(50, dtype=np.float32)
        tm = _RT.from_numpy(matrix)
        tr = _RT.from_numpy(row)
        out = broadcast_add(tm, tr).numpy()
        np.testing.assert_allclose(out, matrix + row, rtol=1e-5)


class TestScalarMul:
    def test_basic(self):
        data = np.array([2, 4, 6], dtype=np.float32)
        t = _RT.from_numpy(data)
        out = scalar_mul(t, 0.5).numpy()
        np.testing.assert_allclose(out, [1, 2, 3])

    def test_zero(self):
        t = _RT.ones([10])
        out = scalar_mul(t, 0.0).numpy()
        np.testing.assert_array_equal(out, 0.0)
