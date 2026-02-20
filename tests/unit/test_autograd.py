"""Comprehensive tests for the autograd engine."""

import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../build"))

from cuDeep.tensor import Tensor, mse_loss, cross_entropy_loss, broadcast_add, softmax
from cuDeep import nn, optim
from cuDeep._core import DType


def numerical_grad(fn, x, eps=1e-4):
    """Finite-difference gradient check."""
    x_np = x.numpy().copy()
    grad = np.zeros_like(x_np)
    it = np.nditer(x_np, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        old = x_np[idx]
        x_np[idx] = old + eps
        fp = fn(Tensor.from_numpy(x_np)).item()
        x_np[idx] = old - eps
        fm = fn(Tensor.from_numpy(x_np)).item()
        grad[idx] = (fp - fm) / (2 * eps)
        x_np[idx] = old
        it.iternext()
    return grad


class TestBasicOps:
    def test_add_grad(self):
        a = Tensor.from_numpy(np.array([1, 2, 3], dtype=np.float32), requires_grad=True)
        b = Tensor.from_numpy(np.array([4, 5, 6], dtype=np.float32), requires_grad=True)
        c = (a + b).sum()
        c.backward()
        np.testing.assert_allclose(a.grad.numpy(), [1, 1, 1])
        np.testing.assert_allclose(b.grad.numpy(), [1, 1, 1])

    def test_sub_grad(self):
        a = Tensor.from_numpy(np.array([3, 2, 1], dtype=np.float32), requires_grad=True)
        b = Tensor.from_numpy(np.array([1, 1, 1], dtype=np.float32), requires_grad=True)
        c = (a - b).sum()
        c.backward()
        np.testing.assert_allclose(a.grad.numpy(), [1, 1, 1])
        np.testing.assert_allclose(b.grad.numpy(), [-1, -1, -1])

    def test_mul_grad(self):
        a = Tensor.from_numpy(np.array([2, 3], dtype=np.float32), requires_grad=True)
        b = Tensor.from_numpy(np.array([4, 5], dtype=np.float32), requires_grad=True)
        c = (a * b).sum()
        c.backward()
        np.testing.assert_allclose(a.grad.numpy(), [4, 5])
        np.testing.assert_allclose(b.grad.numpy(), [2, 3])

    def test_neg_grad(self):
        a = Tensor.from_numpy(np.array([1, -2, 3], dtype=np.float32), requires_grad=True)
        c = (-a).sum()
        c.backward()
        np.testing.assert_allclose(a.grad.numpy(), [-1, -1, -1])

    def test_scalar_mul_grad(self):
        a = Tensor.from_numpy(np.array([1, 2, 3], dtype=np.float32), requires_grad=True)
        c = (a * 3.0).sum()
        c.backward()
        np.testing.assert_allclose(a.grad.numpy(), [3, 3, 3])

    def test_div_grad(self):
        a = Tensor.from_numpy(np.array([6.0, 8.0], dtype=np.float32), requires_grad=True)
        b = Tensor.from_numpy(np.array([2.0, 4.0], dtype=np.float32), requires_grad=True)
        c = (a / b).sum()
        c.backward()
        np.testing.assert_allclose(a.grad.numpy(), [0.5, 0.25], atol=1e-5)
        np.testing.assert_allclose(b.grad.numpy(), [-1.5, -0.5], atol=1e-5)

    def test_pow_grad(self):
        a = Tensor.from_numpy(np.array([2.0, 3.0], dtype=np.float32), requires_grad=True)
        c = (a ** 2).sum()
        c.backward()
        np.testing.assert_allclose(a.grad.numpy(), [4.0, 6.0], atol=1e-4)


class TestMatmulGrad:
    def test_matmul_grad(self):
        a = Tensor.from_numpy(np.array([[1, 2], [3, 4]], dtype=np.float32), requires_grad=True)
        b = Tensor.from_numpy(np.array([[5, 6], [7, 8]], dtype=np.float32), requires_grad=True)
        c = a.matmul(b).sum()
        c.backward()
        expected_ga = np.array([[11, 15], [11, 15]], dtype=np.float32)
        expected_gb = np.array([[4, 4], [6, 6]], dtype=np.float32)
        np.testing.assert_allclose(a.grad.numpy(), expected_ga, atol=1e-4)
        np.testing.assert_allclose(b.grad.numpy(), expected_gb, atol=1e-4)

    def test_matmul_nonsquare(self):
        a = Tensor.from_numpy(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
        b = Tensor.from_numpy(np.random.randn(4, 2).astype(np.float32), requires_grad=True)
        c = a.matmul(b).sum()
        c.backward()
        assert a.grad.shape() == [3, 4]
        assert b.grad.shape() == [4, 2]


class TestUnaryGrad:
    def test_exp_grad(self):
        a = Tensor.from_numpy(np.array([0.0, 1.0], dtype=np.float32), requires_grad=True)
        c = a.exp().sum()
        c.backward()
        expected = np.exp([0.0, 1.0])
        np.testing.assert_allclose(a.grad.numpy(), expected, atol=1e-5)

    def test_log_grad(self):
        a = Tensor.from_numpy(np.array([1.0, 2.0], dtype=np.float32), requires_grad=True)
        c = a.log().sum()
        c.backward()
        np.testing.assert_allclose(a.grad.numpy(), [1.0, 0.5], atol=1e-5)

    def test_sqrt_grad(self):
        a = Tensor.from_numpy(np.array([4.0, 9.0], dtype=np.float32), requires_grad=True)
        c = a.sqrt().sum()
        c.backward()
        np.testing.assert_allclose(a.grad.numpy(), [0.25, 1.0 / 6.0], atol=1e-4)


class TestActivationGrad:
    def test_relu_grad(self):
        a = Tensor.from_numpy(np.array([-1, 0, 1, 2], dtype=np.float32), requires_grad=True)
        c = a.relu().sum()
        c.backward()
        np.testing.assert_allclose(a.grad.numpy(), [0, 0, 1, 1], atol=1e-5)

    def test_sigmoid_grad_numerical(self):
        x_np = np.array([0.5, -0.3, 1.0], dtype=np.float32)
        a = Tensor.from_numpy(x_np, requires_grad=True)
        c = a.sigmoid().sum()
        c.backward()

        def fn(t): return t.sigmoid().sum()
        ng = numerical_grad(fn, Tensor.from_numpy(x_np))
        np.testing.assert_allclose(a.grad.numpy(), ng, atol=1e-3)

    def test_tanh_grad_numerical(self):
        x_np = np.array([0.2, -0.5], dtype=np.float32)
        a = Tensor.from_numpy(x_np, requires_grad=True)
        c = a.tanh().sum()
        c.backward()

        def fn(t): return t.tanh().sum()
        ng = numerical_grad(fn, Tensor.from_numpy(x_np))
        np.testing.assert_allclose(a.grad.numpy(), ng, atol=1e-3)


class TestReductionGrad:
    def test_sum_grad(self):
        a = Tensor.from_numpy(np.array([[1, 2], [3, 4]], dtype=np.float32), requires_grad=True)
        c = a.sum()
        c.backward()
        np.testing.assert_allclose(a.grad.numpy(), np.ones((2, 2)))

    def test_mean_grad(self):
        a = Tensor.from_numpy(np.array([[1, 2], [3, 4]], dtype=np.float32), requires_grad=True)
        c = a.mean()
        c.backward()
        np.testing.assert_allclose(a.grad.numpy(), np.full((2, 2), 0.25))


class TestLossGrad:
    def test_mse_loss_grad(self):
        pred = Tensor.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32), requires_grad=True)
        target = Tensor.from_numpy(np.array([1.5, 2.5, 3.5], dtype=np.float32))
        loss = mse_loss(pred, target)
        loss.backward()
        expected = 2.0 * (np.array([1.0, 2.0, 3.0]) - np.array([1.5, 2.5, 3.5])) / 3.0
        np.testing.assert_allclose(pred.grad.numpy(), expected, atol=1e-5)

    def test_cross_entropy_loss_grad_shape(self):
        logits = Tensor.from_numpy(np.array([[2.0, 1.0, 0.1]], dtype=np.float32), requires_grad=True)
        targets = np.array([0], dtype=np.int32)
        loss = cross_entropy_loss(logits, targets)
        loss.backward()
        assert logits.grad.shape() == [1, 3]


class TestBroadcastAddGrad:
    def test_broadcast_add_grad(self):
        mat = Tensor.from_numpy(np.ones((3, 4), dtype=np.float32), requires_grad=True)
        row = Tensor.from_numpy(np.array([1, 2, 3, 4], dtype=np.float32), requires_grad=True)
        c = broadcast_add(mat, row).sum()
        c.backward()
        np.testing.assert_allclose(mat.grad.numpy(), np.ones((3, 4)))
        np.testing.assert_allclose(row.grad.numpy(), [3, 3, 3, 3])


class TestChainRule:
    def test_linear_chain(self):
        a = Tensor.from_numpy(np.array([2.0], dtype=np.float32), requires_grad=True)
        b = a * 3.0
        c = b + 1.0
        d = c * 2.0
        e = d.sum()
        e.backward()
        np.testing.assert_allclose(a.grad.numpy(), [6.0])

    def test_mlp_chain(self):
        np.random.seed(42)
        x = Tensor.from_numpy(np.random.randn(2, 3).astype(np.float32))
        model = nn.Sequential(
            nn.Linear(3, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
        )
        y = Tensor.from_numpy(np.ones((2, 1), dtype=np.float32))
        loss = mse_loss(model(x), y)
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None, f"grad missing for param shape {p.shape()}"

    def test_grad_accumulation(self):
        w = Tensor.from_numpy(np.array([[1.0, 2.0]], dtype=np.float32), requires_grad=True)
        x1 = Tensor.from_numpy(np.array([[1.0, 0.0]], dtype=np.float32))
        x2 = Tensor.from_numpy(np.array([[0.0, 1.0]], dtype=np.float32))
        y1 = w.matmul(x1.t()).sum()
        y1.backward()
        g1 = w.grad.numpy().copy()
        w.grad = None
        y2 = w.matmul(x2.t()).sum()
        y2.backward()
        g2 = w.grad.numpy().copy()
        assert not np.allclose(g1, g2)


class TestOptimizer:
    def test_sgd_step(self):
        p = Tensor.from_numpy(np.array([10.0], dtype=np.float32), requires_grad=True)
        from cuDeep.nn import Parameter
        param = Parameter(p)
        opt = optim.SGD([param], lr=0.1)
        param.grad = Tensor.from_numpy(np.array([1.0], dtype=np.float32))
        opt.step()
        np.testing.assert_allclose(param.numpy(), [9.9], atol=1e-5)

    def test_adam_step(self):
        p = Tensor.from_numpy(np.array([10.0], dtype=np.float32), requires_grad=True)
        from cuDeep.nn import Parameter
        param = Parameter(p)
        opt = optim.Adam([param], lr=0.1)
        param.grad = Tensor.from_numpy(np.array([1.0], dtype=np.float32))
        opt.step()
        assert param.numpy()[0] < 10.0

    def test_training_loop(self):
        np.random.seed(0)
        model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))
        opt = optim.Adam(model.parameters(), lr=0.01)
        x = Tensor.from_numpy(np.array([[1, 0], [0, 1]], dtype=np.float32))
        y = Tensor.from_numpy(np.array([[1], [0]], dtype=np.float32))

        losses = []
        for _ in range(50):
            opt.zero_grad()
            loss = mse_loss(model(x), y)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], "Loss should decrease during training"


class TestModule:
    def test_state_dict(self):
        model = nn.Linear(3, 2)
        sd = model.state_dict()
        assert "weight" in sd
        assert "bias" in sd
        assert sd["weight"].shape == (2, 3)

    def test_zero_grad(self):
        model = nn.Linear(3, 2)
        x = Tensor.from_numpy(np.random.randn(1, 3).astype(np.float32))
        loss = model(x).sum()
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None
        model.zero_grad()
        for p in model.parameters():
            assert p.grad is None

    def test_parameters_count(self):
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )
        params = model.parameters()
        total = sum(p.numel() for p in params)
        assert total == 10 * 20 + 20 + 20 * 5 + 5
