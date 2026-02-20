"""Tests for cuDeep optimizers — verify actual parameter updates."""

import numpy as np
import pytest

from cuDeep import Tensor, DType
from cuDeep.optim import SGD, Adam, AdamW


class TestSGD:
    def test_basic_step(self):
        """param = param - lr * grad"""
        param = Tensor.from_numpy(np.array([10.0, 20.0], dtype=np.float32))
        grad = Tensor.from_numpy(np.array([1.0, 2.0], dtype=np.float32))
        param.grad = grad
        opt = SGD([param], lr=0.1)
        opt.step()

        result = param.numpy()
        np.testing.assert_allclose(result, [9.9, 19.8], rtol=1e-5)

    def test_multiple_steps(self):
        param = Tensor.from_numpy(np.array([100.0], dtype=np.float32))
        grad = Tensor.from_numpy(np.array([10.0], dtype=np.float32))
        opt = SGD([param], lr=1.0)

        param.grad = grad
        opt.step()
        np.testing.assert_allclose(param.numpy(), [90.0], rtol=1e-5)

        param.grad = grad
        opt.step()
        np.testing.assert_allclose(param.numpy(), [80.0], rtol=1e-5)

    def test_momentum(self):
        param = Tensor.from_numpy(np.array([0.0], dtype=np.float32))
        grad = Tensor.from_numpy(np.array([1.0], dtype=np.float32))
        opt = SGD([param], lr=0.1, momentum=0.9)

        param.grad = grad
        opt.step()
        v1 = param.numpy()[0]

        param.grad = grad
        opt.step()
        v2 = param.numpy()[0]

        assert v2 < v1  # should accelerate

    def test_weight_decay(self):
        param = Tensor.from_numpy(np.array([10.0], dtype=np.float32))
        grad = Tensor.from_numpy(np.array([0.0], dtype=np.float32))
        param.grad = grad
        opt = SGD([param], lr=0.1, weight_decay=0.1)
        opt.step()
        result = param.numpy()
        # grad_effective = 0 + 0.1 * 10 = 1.0
        # param = 10 - 0.1 * 1.0 = 9.9
        np.testing.assert_allclose(result, [9.9], rtol=1e-5)

    def test_zero_grad(self):
        param = Tensor.ones([5])
        param.grad = Tensor.ones([5])
        opt = SGD([param], lr=0.1)
        opt.zero_grad()
        assert param.grad is None

    def test_multiple_params(self):
        p1 = Tensor.from_numpy(np.array([1.0, 2.0], dtype=np.float32))
        p2 = Tensor.from_numpy(np.array([3.0], dtype=np.float32))
        g1 = Tensor.from_numpy(np.array([0.1, 0.2], dtype=np.float32))
        g2 = Tensor.from_numpy(np.array([0.3], dtype=np.float32))
        p1.grad = g1
        p2.grad = g2
        opt = SGD([p1, p2], lr=1.0)
        opt.step()

        np.testing.assert_allclose(p1.numpy(), [0.9, 1.8], rtol=1e-5)
        np.testing.assert_allclose(p2.numpy(), [2.7], rtol=1e-5)


class TestAdam:
    def test_basic_step(self):
        param = Tensor.from_numpy(np.array([10.0], dtype=np.float32))
        grad = Tensor.from_numpy(np.array([1.0], dtype=np.float32))
        opt = Adam([param], lr=0.01)

        initial = param.numpy()[0]
        param.grad = grad
        opt.step()
        updated = param.numpy()[0]

        assert updated < initial

    def test_convergence(self):
        """Repeatedly applying gradient should move param toward zero."""
        param = Tensor.from_numpy(np.array([5.0], dtype=np.float32))
        opt = Adam([param], lr=0.1)

        for _ in range(50):
            param.grad = Tensor.from_numpy(param.numpy().copy())
            opt.step()

        assert abs(param.numpy()[0]) < 1.0

    def test_multiple_params(self):
        p1 = Tensor.from_numpy(np.array([10.0], dtype=np.float32))
        p2 = Tensor.from_numpy(np.array([20.0], dtype=np.float32))
        g1 = Tensor.from_numpy(np.array([1.0], dtype=np.float32))
        g2 = Tensor.from_numpy(np.array([2.0], dtype=np.float32))
        p1.grad = g1
        p2.grad = g2
        opt = Adam([p1, p2], lr=0.01)
        opt.step()

        assert p1.numpy()[0] < 10.0
        assert p2.numpy()[0] < 20.0

    def test_step_count_increments(self):
        param = Tensor.from_numpy(np.array([1.0], dtype=np.float32))
        grad = Tensor.from_numpy(np.array([0.1], dtype=np.float32))
        opt = Adam([param], lr=0.001)
        param.grad = grad
        opt.step()
        assert opt._step_count == 1
        param.grad = grad
        opt.step()
        assert opt._step_count == 2


class TestAdamW:
    def test_basic_step(self):
        param = Tensor.from_numpy(np.array([10.0], dtype=np.float32))
        grad = Tensor.from_numpy(np.array([1.0], dtype=np.float32))
        opt = AdamW([param], lr=0.01, weight_decay=0.01)

        initial = param.numpy()[0]
        param.grad = grad
        opt.step()
        updated = param.numpy()[0]

        assert updated < initial

    def test_weight_decay_effect(self):
        """With zero grad, weight decay alone should shrink params."""
        p_wd = Tensor.from_numpy(np.array([10.0], dtype=np.float32))
        p_no = Tensor.from_numpy(np.array([10.0], dtype=np.float32))
        g = Tensor.from_numpy(np.array([0.0], dtype=np.float32))

        opt_wd = AdamW([p_wd], lr=0.1, weight_decay=0.1)
        opt_no = AdamW([p_no], lr=0.1, weight_decay=0.0)

        p_wd.grad = g
        p_no.grad = g
        opt_wd.step()
        opt_no.step()

        assert p_wd.numpy()[0] < p_no.numpy()[0]
