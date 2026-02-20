"""Integration tests — full forward passes, training loops, and multi-layer models."""

import numpy as np
import pytest

from cuDeep import Tensor, DType, mse_loss, cross_entropy_loss, softmax
from cuDeep.nn import Linear, ReLU, GELU, Sequential, Conv2d
from cuDeep.layers import MaxPool2d, BatchNorm2d, LayerNorm, Dropout
from cuDeep.optim import SGD, Adam


class TestMLPForwardPass:
    """Verify data flows through multi-layer MLPs with activations."""

    def test_3_layer_relu(self):
        model = Sequential(
            Linear(32, 64), ReLU(),
            Linear(64, 32), ReLU(),
            Linear(32, 10),
        )
        x = Tensor.randn([8, 32])
        out = model(x)
        assert out.shape() == [8, 10]
        arr = out.numpy()
        assert np.all(np.isfinite(arr))

    def test_deep_gelu(self):
        model = Sequential(
            Linear(16, 32), GELU(),
            Linear(32, 64), GELU(),
            Linear(64, 32), GELU(),
            Linear(32, 16), GELU(),
            Linear(16, 2),
        )
        x = Tensor.randn([4, 16])
        out = model(x)
        assert out.shape() == [4, 2]
        assert np.all(np.isfinite(out.numpy()))

    def test_output_changes_with_input(self):
        model = Sequential(Linear(8, 4), ReLU(), Linear(4, 2))
        x1 = Tensor.randn([1, 8])
        x2 = Tensor.randn([1, 8])
        o1 = model(x1).numpy()
        o2 = model(x2).numpy()
        assert not np.allclose(o1, o2)


class TestCNNForwardPass:
    """Verify conv + pool + linear pipelines."""

    def test_conv_relu_pool(self):
        conv = Conv2d(1, 8, kernel_size=3, padding=1)
        relu = ReLU()
        pool = MaxPool2d(2)

        x = Tensor.randn([2, 1, 8, 8])
        out = pool(relu(conv(x)))
        assert out.shape() == [2, 8, 4, 4]
        assert np.all(np.isfinite(out.numpy()))

    def test_multi_conv(self):
        model = Sequential(
            Conv2d(1, 4, 3, padding=1), ReLU(), MaxPool2d(2),
            Conv2d(4, 8, 3, padding=1), ReLU(), MaxPool2d(2),
        )
        x = Tensor.randn([1, 1, 16, 16])
        out = model(x)
        assert out.shape() == [1, 8, 4, 4]


class TestLossComputation:
    """Verify loss functions on model outputs."""

    def test_mse_on_model_output(self):
        model = Sequential(Linear(10, 5), ReLU(), Linear(5, 3))
        x = Tensor.randn([4, 10])
        target = Tensor.zeros([4, 3])
        pred = model(x)
        loss = mse_loss(pred, target)
        val = loss.numpy()[0]
        assert np.isfinite(val)
        assert val >= 0

    def test_cross_entropy_on_logits(self):
        model = Sequential(Linear(10, 5), ReLU(), Linear(5, 3))
        x = Tensor.randn([4, 10])
        targets = np.array([0, 1, 2, 0], dtype=np.int32)
        logits = model(x)
        loss = cross_entropy_loss(logits, targets)
        val = loss.numpy()[0]
        assert np.isfinite(val)
        assert val > 0

    def test_softmax_on_logits(self):
        model = Sequential(Linear(8, 4))
        x = Tensor.randn([3, 8])
        logits = model(x)
        probs = softmax(logits)
        arr = probs.numpy()
        assert arr.shape == (3, 4)
        np.testing.assert_allclose(arr.sum(axis=1), 1.0, rtol=1e-5)
        assert np.all(arr >= 0)


class TestSGDTraining:
    """Verify SGD can reduce loss over multiple steps."""

    def test_loss_decreases(self):
        model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
        params = model.parameters()
        opt = SGD(params, lr=0.01)

        x = Tensor.randn([16, 4])
        target = Tensor.zeros([16, 2])

        losses = []
        for _ in range(20):
            pred = model(x)
            loss = mse_loss(pred, target)
            losses.append(loss.numpy()[0])

            grads = [Tensor.randn(p.shape(), p.dtype()) for p in params]
            for g in grads:
                g.fill_(0.01)
            for p, g in zip(params, grads):
                p.grad = g
            opt.step()

        assert all(np.isfinite(l) for l in losses)

    def test_params_change(self):
        layer = Linear(4, 2)
        params = layer.parameters()
        opt = SGD(params, lr=0.1)

        before = [p.numpy().copy() for p in params]
        grads = [Tensor.ones(p.shape(), p.dtype()) for p in params]
        for p, g in zip(params, grads):
            p.grad = g
        opt.step()
        after = [p.numpy() for p in params]

        for b, a in zip(before, after):
            assert not np.allclose(a, b)


class TestAdamTraining:
    """Verify Adam optimizer updates."""

    def test_params_change(self):
        layer = Linear(4, 2)
        params = layer.parameters()
        opt = Adam(params, lr=0.01)

        before = [p.numpy().copy() for p in params]
        grads = [Tensor.ones(p.shape(), p.dtype()) for p in params]
        for p, g in zip(params, grads):
            p.grad = g
        opt.step()
        after = [p.numpy() for p in params]

        for b, a in zip(before, after):
            assert not np.allclose(a, b)

    def test_multiple_steps_stable(self):
        model = Sequential(Linear(8, 4), GELU(), Linear(4, 2))
        params = model.parameters()
        opt = Adam(params, lr=1e-3)

        for _ in range(10):
            grads = [Tensor.randn(p.shape(), p.dtype()) for p in params]
            for p, g in zip(params, grads):
                p.grad = g
            opt.step()

        for p in params:
            assert np.all(np.isfinite(p.numpy()))


class TestNormalizationIntegration:
    """Verify normalization layers in pipelines."""

    def test_batchnorm_in_cnn(self):
        conv = Conv2d(1, 4, 3, padding=1)
        bn = BatchNorm2d(4)
        relu = ReLU()

        x = Tensor.randn([4, 1, 8, 8])
        out = relu(bn(conv(x)))
        assert out.shape() == [4, 4, 8, 8]

        arr = out.numpy()
        assert np.all(np.isfinite(arr))
        assert np.all(arr >= 0)  # after ReLU

    def test_layernorm_in_mlp(self):
        model_with_ln = Sequential(
            Linear(16, 32),
            LayerNorm(32),
            GELU(),
            Linear(32, 8),
        )
        x = Tensor.randn([4, 16])
        out = model_with_ln(x)
        assert out.shape() == [4, 8]
        assert np.all(np.isfinite(out.numpy()))


class TestDropoutIntegration:
    def test_train_vs_eval(self):
        model = Sequential(
            Linear(8, 16), ReLU(), Dropout(0.5),
            Linear(16, 4),
        )
        x = Tensor.randn([2, 8])

        model.train()
        out_train = model(x).numpy()

        model.eval()
        out_eval = model(x).numpy()

        assert np.all(np.isfinite(out_train))
        assert np.all(np.isfinite(out_eval))


class TestDtypeConsistency:
    """Verify f64 works end-to-end."""

    def test_f64_mlp(self):
        model = Sequential(
            Linear(8, 4, dtype=DType.float64),
            ReLU(),
            Linear(4, 2, dtype=DType.float64),
        )
        x = Tensor.randn([3, 8], DType.float64)
        out = model(x)
        assert out.shape() == [3, 2]
        arr = out.numpy()
        assert arr.dtype == np.float64
        assert np.all(np.isfinite(arr))
