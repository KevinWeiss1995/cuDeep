"""Autograd-aware Tensor that wraps the native C++ Tensor.

All arithmetic, matmul, and activation operations build a computation graph
that supports reverse-mode automatic differentiation via ``.backward()``.
"""

from __future__ import annotations
from typing import Optional, Sequence, Union
import numpy as np

from cuDeep._core import (
    Tensor as _RawTensor,
    DType,
    # elementwise
    scalar_mul as _scalar_mul,
    broadcast_add as _broadcast_add,
    div_op as _div_op,
    neg as _neg,
    # unary math
    exp_op as _exp_op,
    log_op as _log_op,
    sqrt_op as _sqrt_op,
    pow_op as _pow_op,
    abs_op as _abs_op,
    clamp_op as _clamp_op,
    gt_mask as _gt_mask,
    # reductions
    sum as _sum,
    mean as _mean,
    max as _max,
    min as _min,
    sum_reduce_rows as _sum_reduce_rows,
    # activations
    relu as _relu,
    sigmoid as _sigmoid,
    tanh_act as _tanh_act,
    gelu as _gelu,
    silu as _silu,
    leaky_relu as _leaky_relu,
    activation_backward as _activation_backward,
    # softmax / loss
    softmax as _softmax,
    mse_loss as _mse_loss,
    cross_entropy_loss as _cross_entropy_loss,
    # conv / pool / norm
    conv2d_forward as _conv2d_forward,
    conv2d_backward_data as _conv2d_backward_data,
    conv2d_backward_weight as _conv2d_backward_weight,
    max_pool2d as _max_pool2d,
    avg_pool2d as _avg_pool2d,
    batchnorm_forward as _batchnorm_forward,
    layernorm_forward as _layernorm_forward,
)
from cuDeep.autograd import Function, Context, backward as _backward


class Tensor:
    """Autograd-enabled tensor backed by GPU memory.

    Wraps the C++ ``_RawTensor`` and records operations into a DAG so that
    ``loss.backward()`` computes gradients automatically.
    """

    __slots__ = ("_data", "grad", "requires_grad", "_grad_fn", "_ctx", "_prev")

    def __init__(self, data, requires_grad=False, dtype=None):
        if isinstance(data, _RawTensor):
            self._data = data
        elif isinstance(data, np.ndarray):
            arr = np.ascontiguousarray(data)
            if dtype is not None:
                np_dt = np.float64 if dtype == DType.float64 else np.float32
                arr = arr.astype(np_dt, copy=False)
            elif arr.dtype not in (np.float32, np.float64):
                arr = arr.astype(np.float32)
            self._data = _RawTensor.from_numpy(arr)
        elif isinstance(data, (list, tuple, int, float)):
            arr = np.array(data, dtype=np.float32)
            self._data = _RawTensor.from_numpy(arr)
        else:
            raise TypeError(f"Cannot create Tensor from {type(data)}")
        self.grad: Optional[Tensor] = None
        self.requires_grad = requires_grad
        self._grad_fn = None
        self._ctx = None
        self._prev: tuple = ()

    @classmethod
    def _wrap(cls, raw: _RawTensor, requires_grad=False) -> Tensor:
        t = object.__new__(cls)
        t._data = raw
        t.grad = None
        t.requires_grad = requires_grad
        t._grad_fn = None
        t._ctx = None
        t._prev = ()
        return t

    # ---- Factory methods ----

    @staticmethod
    def zeros(shape, dtype=DType.float32, requires_grad=False) -> Tensor:
        return Tensor._wrap(_RawTensor.zeros(list(shape), dtype), requires_grad)

    @staticmethod
    def ones(shape, dtype=DType.float32, requires_grad=False) -> Tensor:
        return Tensor._wrap(_RawTensor.ones(list(shape), dtype), requires_grad)

    @staticmethod
    def randn(shape, dtype=DType.float32, requires_grad=False) -> Tensor:
        return Tensor._wrap(_RawTensor.randn(list(shape), dtype), requires_grad)

    @staticmethod
    def from_numpy(arr, requires_grad=False) -> Tensor:
        arr = np.ascontiguousarray(arr)
        if arr.dtype not in (np.float32, np.float64):
            arr = arr.astype(np.float32)
        return Tensor._wrap(_RawTensor.from_numpy(arr), requires_grad)

    @staticmethod
    def full(shape, value, dtype=DType.float32, requires_grad=False) -> Tensor:
        t = Tensor._wrap(_RawTensor(list(shape), dtype), requires_grad)
        t._data.fill_(float(value))
        return t

    # ---- Properties ----

    def shape(self):
        return self._data.shape()

    def dtype(self):
        return self._data.dtype()

    def ndim(self):
        return self._data.ndim()

    def numel(self):
        return self._data.numel()

    def is_contiguous(self):
        return self._data.is_contiguous()

    def numpy(self):
        return self._data.numpy()

    def item(self):
        assert self._data.numel() == 1, "item() only for single-element tensors"
        return self._data.numpy().item()

    def detach(self) -> Tensor:
        return Tensor._wrap(self._data, requires_grad=False)

    def contiguous(self) -> Tensor:
        if self._data.is_contiguous():
            return self
        return Tensor._wrap(self._data.contiguous(), self.requires_grad)

    def clone(self) -> Tensor:
        raw_copy = _RawTensor(self._data.shape(), self._data.dtype())
        from cuDeep._core import Tensor as _RT
        raw_np = self._data.numpy()
        raw_copy = _RawTensor.from_numpy(raw_np)
        return Tensor._wrap(raw_copy, self.requires_grad)

    # ---- In-place ops (no autograd) ----

    def fill_(self, value):
        self._data.fill_(float(value))
        return self

    def zero_(self):
        self._data.zero_()
        return self

    # ---- Reshape / transpose ----

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = list(shape[0])
        else:
            shape = list(shape)
        return Tensor._wrap(self._data.reshape(shape), self.requires_grad)

    def transpose(self, dim0, dim1):
        return TransposeOp.apply(self, dim0, dim1)

    def t(self):
        return self.transpose(0, 1)

    # ---- Elementwise arithmetic (with autograd) ----

    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor.full(self.shape(), other, self.dtype())
        return AddOp.apply(self, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor.full(self.shape(), other, self.dtype())
        return SubOp.apply(self, other)

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor.full(self.shape(), other, self.dtype())
        return SubOp.apply(other, self)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return ScalarMulOp.apply(self, float(other))
        return MulOp.apply(self, other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return ScalarMulOp.apply(self, 1.0 / float(other))
        return DivOp.apply(self, other)

    def __neg__(self):
        return NegOp.apply(self)

    def __pow__(self, exponent):
        return PowOp.apply(self, float(exponent))

    # ---- Matmul ----

    def matmul(self, other):
        return MatMulOp.apply(self, other)

    def __matmul__(self, other):
        return self.matmul(other)

    # ---- Reductions ----

    def sum(self):
        return SumOp.apply(self)

    def mean(self):
        return MeanOp.apply(self)

    # ---- Unary math ----

    def exp(self):
        return ExpOp.apply(self)

    def log(self):
        return LogOp.apply(self)

    def sqrt(self):
        return SqrtOp.apply(self)

    def abs(self):
        return AbsOp.apply(self)

    def clamp(self, lo, hi):
        return ClampOp.apply(self, float(lo), float(hi))

    # ---- Activations (with autograd) ----

    def relu(self):
        return ReLUOp.apply(self)

    def sigmoid(self):
        return SigmoidOp.apply(self)

    def tanh(self):
        return TanhOp.apply(self)

    def gelu(self):
        return GELUOp.apply(self)

    def silu(self):
        return SiLUOp.apply(self)

    # ---- Backward ----

    def backward(self, grad_output=None):
        _backward(self, grad_output)

    # ---- Repr ----

    def __repr__(self):
        n = self.numel()
        if n <= 10:
            data_str = repr(self.numpy())
        elif n <= 100:
            data_str = f"shape={self.shape()}"
        else:
            data_str = f"shape={self.shape()}"
        s = f"Tensor({data_str}, dtype={self.dtype()}"
        if self.requires_grad:
            s += ", requires_grad=True"
            if self._grad_fn is not None:
                s += f", grad_fn={self._grad_fn.__name__}"
        return s + ")"

    def __len__(self):
        return self.shape()[0]

    def __bool__(self):
        if self.numel() != 1:
            raise RuntimeError("bool() on multi-element tensor is ambiguous")
        return bool(self.item())


# ---- Convenience functions ----

def zeros_like(t: Tensor) -> Tensor:
    return Tensor.zeros(t.shape(), t.dtype())

def ones_like(t: Tensor) -> Tensor:
    return Tensor.ones(t.shape(), t.dtype())

def randn_like(t: Tensor) -> Tensor:
    return Tensor.randn(t.shape(), t.dtype())


# ---------------------------------------------------------------------------
# Autograd Functions
# ---------------------------------------------------------------------------

def _ensure_raw(x):
    """Extract C++ _RawTensor from a Tensor, or return as-is."""
    return x._data if isinstance(x, Tensor) else x


class AddOp(Function):
    @staticmethod
    def forward(ctx, a, b):
        ad, bd = a._data.contiguous() if not a._data.is_contiguous() else a._data, \
                 b._data.contiguous() if not b._data.is_contiguous() else b._data
        ctx.save_for_backward(a, b)
        return Tensor._wrap(ad + bd)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        ga = grad_output if a.requires_grad else None
        gb = grad_output if b.requires_grad else None
        return ga, gb


class SubOp(Function):
    @staticmethod
    def forward(ctx, a, b):
        ad = a._data.contiguous() if not a._data.is_contiguous() else a._data
        bd = b._data.contiguous() if not b._data.is_contiguous() else b._data
        ctx.save_for_backward(a, b)
        return Tensor._wrap(ad - bd)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        ga = grad_output if a.requires_grad else None
        gb = Tensor._wrap(_neg(grad_output._data)) if b.requires_grad else None
        return ga, gb


class MulOp(Function):
    @staticmethod
    def forward(ctx, a, b):
        ad = a._data.contiguous() if not a._data.is_contiguous() else a._data
        bd = b._data.contiguous() if not b._data.is_contiguous() else b._data
        ctx.save_for_backward(a, b)
        return Tensor._wrap(ad * bd)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        gd = grad_output._data
        ga = Tensor._wrap(gd * b._data) if a.requires_grad else None
        gb = Tensor._wrap(gd * a._data) if b.requires_grad else None
        return ga, gb


class DivOp(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return Tensor._wrap(_div_op(a._data, b._data))

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        gd = grad_output._data
        ga = Tensor._wrap(_div_op(gd, b._data)) if a.requires_grad else None
        gb = None
        if b.requires_grad:
            neg_a_over_b2 = _neg(_div_op(a._data, b._data * b._data))
            gb = Tensor._wrap(gd * neg_a_over_b2)
        return ga, gb


class ScalarMulOp(Function):
    @staticmethod
    def forward(ctx, a, scalar):
        ctx.save_for_backward(a, scalar)
        return Tensor._wrap(_scalar_mul(a._data, scalar))

    @staticmethod
    def backward(ctx, grad_output):
        a, scalar = ctx.saved_tensors
        ga = Tensor._wrap(_scalar_mul(grad_output._data, scalar)) if a.requires_grad else None
        return (ga,)


class NegOp(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return Tensor._wrap(_neg(a._data))

    @staticmethod
    def backward(ctx, grad_output):
        return (Tensor._wrap(_neg(grad_output._data)),)


class MatMulOp(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        ad = a._data.contiguous() if not a._data.is_contiguous() else a._data
        bd = b._data.contiguous() if not b._data.is_contiguous() else b._data
        return Tensor._wrap(ad.matmul(bd))

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        gd = grad_output._data
        ga = gb = None
        bd = b._data.contiguous() if not b._data.is_contiguous() else b._data
        ad = a._data.contiguous() if not a._data.is_contiguous() else a._data
        if a.requires_grad:
            bt = bd.transpose(0, 1).contiguous()
            ga = Tensor._wrap(gd.matmul(bt))
        if b.requires_grad:
            at = ad.transpose(0, 1).contiguous()
            gb = Tensor._wrap(at.matmul(gd))
        return ga, gb


class SumOp(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return Tensor._wrap(_sum(a._data))

    @staticmethod
    def backward(ctx, grad_output):
        (a,) = ctx.saved_tensors
        val = grad_output.item()
        g = Tensor.full(a.shape(), val, a.dtype())
        return (g,)


class MeanOp(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return Tensor._wrap(_mean(a._data))

    @staticmethod
    def backward(ctx, grad_output):
        (a,) = ctx.saved_tensors
        n = a.numel()
        val = grad_output.item() / n
        g = Tensor.full(a.shape(), val, a.dtype())
        return (g,)


class TransposeOp(Function):
    @staticmethod
    def forward(ctx, a, dim0, dim1):
        ctx.save_for_backward(dim0, dim1)
        return Tensor._wrap(a._data.transpose(dim0, dim1), a.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        dim0, dim1 = ctx.saved_tensors
        return (Tensor._wrap(grad_output._data.transpose(dim0, dim1)),)


class ExpOp(Function):
    @staticmethod
    def forward(ctx, a):
        out = _exp_op(a._data)
        ctx.save_for_backward(Tensor._wrap(out))
        return Tensor._wrap(out)

    @staticmethod
    def backward(ctx, grad_output):
        (exp_a,) = ctx.saved_tensors
        return (Tensor._wrap(grad_output._data * exp_a._data),)


class LogOp(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return Tensor._wrap(_log_op(a._data))

    @staticmethod
    def backward(ctx, grad_output):
        (a,) = ctx.saved_tensors
        return (Tensor._wrap(_div_op(grad_output._data, a._data)),)


class SqrtOp(Function):
    @staticmethod
    def forward(ctx, a):
        out = _sqrt_op(a._data)
        ctx.save_for_backward(Tensor._wrap(out))
        return Tensor._wrap(out)

    @staticmethod
    def backward(ctx, grad_output):
        (sqrt_a,) = ctx.saved_tensors
        two_sqrt = _scalar_mul(sqrt_a._data, 2.0)
        return (Tensor._wrap(_div_op(grad_output._data, two_sqrt)),)


class PowOp(Function):
    @staticmethod
    def forward(ctx, a, exponent):
        ctx.save_for_backward(a, exponent)
        return Tensor._wrap(_pow_op(a._data, exponent))

    @staticmethod
    def backward(ctx, grad_output):
        a, exponent = ctx.saved_tensors
        inner = _scalar_mul(_pow_op(a._data, exponent - 1.0), exponent)
        return (Tensor._wrap(grad_output._data * inner),)


class AbsOp(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return Tensor._wrap(_abs_op(a._data))

    @staticmethod
    def backward(ctx, grad_output):
        (a,) = ctx.saved_tensors
        mask_pos = _gt_mask(a._data, 0.0)
        mask_neg = _neg(_gt_mask(_neg(a._data), 0.0))
        sign = mask_pos + mask_neg
        return (Tensor._wrap(grad_output._data * sign),)


class ClampOp(Function):
    @staticmethod
    def forward(ctx, a, lo, hi):
        ctx.save_for_backward(a, lo, hi)
        return Tensor._wrap(_clamp_op(a._data, lo, hi))

    @staticmethod
    def backward(ctx, grad_output):
        a, lo, hi = ctx.saved_tensors
        mask_lo = _gt_mask(a._data, lo)
        mask_hi = _gt_mask(_neg(a._data), -hi)
        mask = mask_lo * mask_hi
        return (Tensor._wrap(grad_output._data * mask),)


# ---- Activation autograd functions ----

def _make_activation_op(act_name):
    class _Op(Function):
        @staticmethod
        def forward(ctx, a):
            fwd_fn = {
                "relu": _relu, "sigmoid": _sigmoid, "tanh": _tanh_act,
                "gelu": _gelu, "silu": _silu, "leaky_relu": _leaky_relu,
            }[act_name]
            ctx.save_for_backward(a)
            return Tensor._wrap(fwd_fn(a._data))

        @staticmethod
        def backward(ctx, grad_output):
            (a,) = ctx.saved_tensors
            return (Tensor._wrap(
                _activation_backward(grad_output._data, a._data, act_name)
            ),)

    _Op.__name__ = f"{act_name.title().replace('_','')}Op"
    return _Op


ReLUOp = _make_activation_op("relu")
SigmoidOp = _make_activation_op("sigmoid")
TanhOp = _make_activation_op("tanh")
GELUOp = _make_activation_op("gelu")
SiLUOp = _make_activation_op("silu")
LeakyReLUOp = _make_activation_op("leaky_relu")


class _LeakyReLUAlphaOp(Function):
    @staticmethod
    def forward(ctx, a, alpha):
        ctx.save_for_backward(a, alpha)
        return Tensor._wrap(_leaky_relu(a._data, alpha))

    @staticmethod
    def backward(ctx, grad_output):
        a, alpha = ctx.saved_tensors
        return (Tensor._wrap(
            _activation_backward(grad_output._data, a._data, "leaky_relu", alpha)
        ),)


# ---- Higher-level autograd functions ----

class BroadcastAddOp(Function):
    """matrix [N,M] + row [M] -> [N,M] with grad for both."""
    @staticmethod
    def forward(ctx, matrix, row):
        ctx.save_for_backward(matrix, row)
        return Tensor._wrap(_broadcast_add(matrix._data, row._data))

    @staticmethod
    def backward(ctx, grad_output):
        matrix, row = ctx.saved_tensors
        g_matrix = grad_output if matrix.requires_grad else None
        g_row = None
        if row.requires_grad:
            g_row = Tensor._wrap(_sum_reduce_rows(grad_output._data))
        return g_matrix, g_row


class SoftmaxOp(Function):
    @staticmethod
    def forward(ctx, a):
        out = _softmax(a._data)
        ctx.save_for_backward(Tensor._wrap(out))
        return Tensor._wrap(out)

    @staticmethod
    def backward(ctx, grad_output):
        (sm,) = ctx.saved_tensors
        g = grad_output._data * sm._data
        row_sums = _sum_reduce_rows(g)
        correction = _broadcast_add(
            _RawTensor.zeros(list(sm._data.shape()), sm._data.dtype()),
            row_sums
        )
        return (Tensor._wrap(g - sm._data * correction),)


class MSELossOp(Function):
    @staticmethod
    def forward(ctx, pred, target):
        ctx.save_for_backward(pred, target)
        return Tensor._wrap(_mse_loss(pred._data, target._data))

    @staticmethod
    def backward(ctx, grad_output):
        pred, target = ctx.saved_tensors
        n = pred.numel()
        diff = pred._data - target._data
        scale = 2.0 / n * grad_output.item()
        g_pred = Tensor._wrap(_scalar_mul(diff, scale))
        g_target = Tensor._wrap(_scalar_mul(diff, -scale))
        return g_pred, g_target


class CrossEntropyLossOp(Function):
    @staticmethod
    def forward(ctx, logits, targets_np):
        ctx.save_for_backward(logits, targets_np)
        return Tensor._wrap(_cross_entropy_loss(logits._data, targets_np))

    @staticmethod
    def backward(ctx, grad_output):
        logits, targets_np = ctx.saved_tensors
        sm = _softmax(logits._data)
        batch = logits.shape()[0]
        num_classes = logits.shape()[1]
        one_hot = np.zeros((batch, num_classes), dtype=np.float32)
        for i, c in enumerate(targets_np):
            one_hot[i, c] = 1.0
        one_hot_raw = _RawTensor.from_numpy(one_hot)
        scale = grad_output.item() / batch
        grad = _scalar_mul(sm - one_hot_raw, scale)
        return (Tensor._wrap(grad),)


class Conv2dOp(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding):
        bias_raw = bias._data if bias is not None else None
        out = _conv2d_forward(
            input._data, weight._data, bias_raw,
            list(stride), list(padding))
        ctx.save_for_backward(input, weight, bias, stride, padding)
        return Tensor._wrap(out)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, stride, padding = ctx.saved_tensors
        g_input = g_weight = g_bias = None
        stride_l = list(stride)
        padding_l = list(padding)

        if input.requires_grad:
            g_input = Tensor._wrap(_conv2d_backward_data(
                grad_output._data, weight._data,
                list(input.shape()), stride_l, padding_l))

        if weight.requires_grad:
            g_weight = Tensor._wrap(_conv2d_backward_weight(
                grad_output._data, input._data,
                list(weight.shape()), stride_l, padding_l))

        if bias is not None and bias.requires_grad:
            go = grad_output._data.numpy()
            g_bias_np = go.sum(axis=(0, 2, 3)).astype(np.float32)
            g_bias = Tensor.from_numpy(g_bias_np)

        return g_input, g_weight, g_bias


# ---- Functional API (mirrors torch.nn.functional) ----

def relu(x):    return x.relu()
def sigmoid(x): return x.sigmoid()
def tanh(x):    return x.tanh()
def gelu(x):    return x.gelu()
def silu(x):    return x.silu()
def softmax(x): return SoftmaxOp.apply(x)
def mse_loss(pred, target): return MSELossOp.apply(pred, target)
def cross_entropy_loss(logits, targets_np):
    return CrossEntropyLossOp.apply(logits, targets_np)
def broadcast_add(matrix, row):
    return BroadcastAddOp.apply(matrix, row)
def conv2d(input, weight, bias=None, stride=(1,1), padding=(0,0)):
    return Conv2dOp.apply(input, weight, bias, stride, padding)
