"""Tests for cuDeep.nn module."""

import numpy as np
import pytest

from cuDeep import Tensor, DType
from cuDeep.nn import Module, Linear, Sequential


class TestModule:
    def test_base_module_forward_raises(self):
        m = Module()
        with pytest.raises(NotImplementedError):
            m.forward()

    def test_parameters_empty(self):
        m = Module()
        assert m.parameters() == []

    def test_train_eval_toggle(self):
        m = Module()
        assert m._training is True
        m.eval()
        assert m._training is False
        m.train()
        assert m._training is True


class TestLinear:
    def test_creation(self):
        layer = Linear(10, 5)
        params = layer.parameters()
        assert len(params) == 2
        weight = layer._parameters["weight"]
        bias = layer._parameters["bias"]
        assert weight.shape() == [5, 10]
        assert bias.shape() == [5]

    def test_creation_no_bias(self):
        layer = Linear(10, 5, bias=False)
        params = layer.parameters()
        assert len(params) == 1

    def test_forward_shape(self):
        layer = Linear(8, 4)
        x = Tensor.randn([2, 8])
        out = layer(x)
        assert out.shape() == [2, 4]

    def test_forward_deterministic(self):
        layer = Linear(3, 2, bias=False)

        W = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        layer._parameters["weight"] = Tensor.from_numpy(W)

        x_data = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.float32)
        x = Tensor.from_numpy(x_data)
        out = layer(x)
        result = out.numpy()
        expected = x_data @ W.T
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_forward_with_bias(self):
        layer = Linear(2, 2)

        W = np.eye(2, dtype=np.float32)
        b = np.array([100, 200], dtype=np.float32)
        layer._parameters["weight"] = Tensor.from_numpy(W)
        layer._parameters["bias"] = Tensor.from_numpy(b)

        x_data = np.array([[1, 2]], dtype=np.float32)
        x = Tensor.from_numpy(x_data)
        out = layer(x)
        result = out.numpy()
        expected = x_data @ W.T + b
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_f64(self):
        layer = Linear(4, 3, dtype=DType.float64)
        x = Tensor.randn([1, 4], DType.float64)
        out = layer(x)
        assert out.shape() == [1, 3]


class TestSequential:
    def test_creation(self):
        model = Sequential(Linear(10, 5), Linear(5, 2))
        params = model.parameters()
        assert len(params) == 4

    def test_forward_shape(self):
        model = Sequential(Linear(10, 5), Linear(5, 2))
        x = Tensor.randn([4, 10])
        out = model(x)
        assert out.shape() == [4, 2]

    def test_train_eval_propagates(self):
        l1 = Linear(10, 5)
        l2 = Linear(5, 2)
        model = Sequential(l1, l2)
        model.eval()
        assert l1._training is False
        assert l2._training is False
        model.train()
        assert l1._training is True
        assert l2._training is True

    def test_deeper_model(self):
        model = Sequential(
            Linear(784, 256),
            Linear(256, 128),
            Linear(128, 10),
        )
        x = Tensor.randn([8, 784])
        out = model(x)
        assert out.shape() == [8, 10]
        params = model.parameters()
        assert len(params) == 6
