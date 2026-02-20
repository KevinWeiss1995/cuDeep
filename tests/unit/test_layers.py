"""Tests for cuDeep.layers — pooling, normalization, dropout."""

import numpy as np
import pytest

from cuDeep import Tensor, DType
from cuDeep.layers import (
    BatchNorm2d, LayerNorm,
    MaxPool2d, AvgPool2d,
    Dropout,
)
from cuDeep.nn import Conv2d, ReLU, Sigmoid, GELU, SiLU, Tanh, LeakyReLU, Sequential


# ---- Activation layers ----

class TestActivationLayers:
    def test_relu_layer(self):
        data = np.array([-2, -1, 0, 1, 2], dtype=np.float32)
        t = Tensor.from_numpy(data)
        out = ReLU()(t).numpy()
        np.testing.assert_allclose(out, np.maximum(data, 0))

    def test_sigmoid_layer(self):
        data = np.array([0.0], dtype=np.float32)
        t = Tensor.from_numpy(data)
        out = Sigmoid()(t).numpy()
        np.testing.assert_allclose(out, [0.5], atol=1e-5)

    def test_gelu_layer(self):
        data = np.array([0.0, 1.0], dtype=np.float32)
        t = Tensor.from_numpy(data)
        out = GELU()(t).numpy()
        assert abs(out[0]) < 1e-5
        assert 0.8 < out[1] < 0.85

    def test_silu_layer(self):
        data = np.array([0.0], dtype=np.float32)
        t = Tensor.from_numpy(data)
        out = SiLU()(t).numpy()
        np.testing.assert_allclose(out, [0.0], atol=1e-5)

    def test_tanh_layer(self):
        data = np.linspace(-2, 2, 10).astype(np.float32)
        t = Tensor.from_numpy(data)
        out = Tanh()(t).numpy()
        np.testing.assert_allclose(out, np.tanh(data), rtol=1e-5)

    def test_leaky_relu_layer(self):
        data = np.array([-10.0, 5.0], dtype=np.float32)
        t = Tensor.from_numpy(data)
        out = LeakyReLU(0.1)(t).numpy()
        np.testing.assert_allclose(out, [-1.0, 5.0], rtol=1e-5)


# ---- Conv2d layer ----

class TestConv2dLayer:
    def test_output_shape(self):
        layer = Conv2d(3, 16, kernel_size=3, padding=1)
        x = Tensor.randn([2, 3, 8, 8])
        out = layer(x)
        assert out.shape() == [2, 16, 8, 8]

    def test_output_shape_no_padding(self):
        layer = Conv2d(1, 4, kernel_size=3)
        x = Tensor.randn([1, 1, 6, 6])
        out = layer(x)
        assert out.shape() == [1, 4, 4, 4]

    def test_stride(self):
        layer = Conv2d(1, 1, kernel_size=3, stride=2, padding=1)
        x = Tensor.randn([1, 1, 8, 8])
        out = layer(x)
        assert out.shape() == [1, 1, 4, 4]


# ---- Pooling layers ----

class TestMaxPool2d:
    def test_output_shape(self):
        pool = MaxPool2d(2)
        x = Tensor.randn([1, 1, 4, 4])
        out = pool(x)
        assert out.shape() == [1, 1, 2, 2]

    def test_values(self):
        data = np.array([[[[1, 2, 3, 4],
                           [5, 6, 7, 8],
                           [9, 10, 11, 12],
                           [13, 14, 15, 16]]]], dtype=np.float32)
        t = Tensor.from_numpy(data)
        pool = MaxPool2d(2)
        out = pool(t).numpy()
        assert out.shape == (1, 1, 2, 2)
        np.testing.assert_allclose(out[0, 0, 0, 0], 6.0)
        np.testing.assert_allclose(out[0, 0, 0, 1], 8.0)
        np.testing.assert_allclose(out[0, 0, 1, 0], 14.0)
        np.testing.assert_allclose(out[0, 0, 1, 1], 16.0)

    def test_stride(self):
        pool = MaxPool2d(3, stride=1, padding=1)
        x = Tensor.randn([1, 2, 5, 5])
        out = pool(x)
        assert out.shape() == [1, 2, 5, 5]


class TestAvgPool2d:
    def test_output_shape(self):
        pool = AvgPool2d(2)
        x = Tensor.randn([1, 1, 4, 4])
        out = pool(x)
        assert out.shape() == [1, 1, 2, 2]

    def test_values(self):
        data = np.ones((1, 1, 4, 4), dtype=np.float32)
        t = Tensor.from_numpy(data)
        pool = AvgPool2d(2)
        out = pool(t).numpy()
        np.testing.assert_allclose(out, 1.0, rtol=1e-5)

    def test_known_values(self):
        data = np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)
        t = Tensor.from_numpy(data)
        pool = AvgPool2d(2)
        out = pool(t).numpy()
        np.testing.assert_allclose(out[0, 0, 0, 0], 2.5, rtol=1e-5)


# ---- BatchNorm2d ----

class TestBatchNorm2d:
    def test_output_shape(self):
        bn = BatchNorm2d(8)
        x = Tensor.randn([4, 8, 3, 3])
        out = bn(x)
        assert out.shape() == [4, 8, 3, 3]

    def test_normalized_output_stats(self):
        bn = BatchNorm2d(2)
        data = np.random.randn(8, 2, 4, 4).astype(np.float32) * 5 + 3
        x = Tensor.from_numpy(data)
        out = bn(x).numpy()
        for c in range(2):
            channel = out[:, c, :, :]
            assert abs(channel.mean()) < 0.5
            assert abs(channel.std() - 1.0) < 0.5

    def test_eval_mode(self):
        bn = BatchNorm2d(2)
        x = Tensor.randn([4, 2, 3, 3])
        bn(x)  # training pass to populate running stats
        bn.eval()
        out = bn(x)
        assert out.shape() == [4, 2, 3, 3]


# ---- LayerNorm ----

class TestLayerNorm:
    def test_output_shape(self):
        ln = LayerNorm(16)
        x = Tensor.randn([4, 16])
        out = ln(x)
        assert out.shape() == [4, 16]

    def test_normalized_stats(self):
        ln = LayerNorm(64)
        data = np.random.randn(8, 64).astype(np.float32) * 10 + 5
        x = Tensor.from_numpy(data)
        out = ln(x).numpy()
        for i in range(8):
            row = out[i]
            assert abs(row.mean()) < 0.3
            assert abs(row.std() - 1.0) < 0.3


# ---- Dropout ----

class TestDropout:
    def test_eval_passthrough(self):
        drop = Dropout(0.5)
        drop.eval()
        data = np.array([1, 2, 3], dtype=np.float32)
        t = Tensor.from_numpy(data)
        out = drop(t).numpy()
        np.testing.assert_array_equal(out, data)

    def test_training_scales(self):
        drop = Dropout(0.5)
        drop.train()
        t = Tensor.ones([10])
        out = drop(t).numpy()
        np.testing.assert_allclose(out, 2.0, rtol=1e-5)

    def test_zero_prob_passthrough(self):
        drop = Dropout(0.0)
        t = Tensor.ones([10])
        out = drop(t).numpy()
        np.testing.assert_allclose(out, 1.0)


# ---- Integration: Sequential with multiple layer types ----

class TestSequentialIntegration:
    def test_mlp_with_activations(self):
        from cuDeep.nn import Linear
        model = Sequential(
            Linear(16, 32),
            ReLU(),
            Linear(32, 8),
            GELU(),
            Linear(8, 2),
        )
        x = Tensor.randn([4, 16])
        out = model(x)
        assert out.shape() == [4, 2]
        assert len(model.parameters()) == 6

    def test_conv_pool_model(self):
        model = Sequential(
            Conv2d(1, 4, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(2),
        )
        x = Tensor.randn([2, 1, 8, 8])
        out = model(x)
        assert out.shape() == [2, 4, 4, 4]
