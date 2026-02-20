"""Reverse-mode automatic differentiation engine for cuDeep.

Implements a tape-based autograd that records operations into a DAG,
then traverses it in reverse topological order to compute gradients.
"""

from __future__ import annotations
from typing import Tuple, Optional, Sequence
import numpy as np

_grad_enabled = True


class no_grad:
    """Context manager that disables gradient tracking.

    Usage::

        with cuDeep.autograd.no_grad():
            pred = model(x)  # no graph built
    """

    def __enter__(self):
        global _grad_enabled
        self._prev = _grad_enabled
        _grad_enabled = False
        return self

    def __exit__(self, *args):
        global _grad_enabled
        _grad_enabled = self._prev


def is_grad_enabled() -> bool:
    return _grad_enabled


class Context:
    """Stores data needed by a Function's backward pass."""

    __slots__ = ("saved_data",)

    def __init__(self):
        self.saved_data: list = []

    def save_for_backward(self, *args):
        self.saved_data = list(args)

    @property
    def saved_tensors(self) -> list:
        return self.saved_data


class Function:
    """Base class for differentiable operations.

    Subclasses implement ``forward`` and ``backward`` as static methods.
    ``apply`` wires them into the autograd graph.
    """

    @staticmethod
    def forward(ctx: Context, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def backward(ctx: Context, grad_output):
        raise NotImplementedError

    @classmethod
    def apply(cls, *args, **kwargs):
        from cuDeep.tensor import Tensor

        ctx = Context()
        tensor_args = [a for a in args if isinstance(a, Tensor)]
        needs_grad = _grad_enabled and any(t.requires_grad for t in tensor_args)

        result = cls.forward(ctx, *args, **kwargs)

        if isinstance(result, Tensor):
            if needs_grad:
                result.requires_grad = True
                result._grad_fn = cls
                result._ctx = ctx
                result._prev = tuple(tensor_args)
            return result

        if isinstance(result, tuple):
            out = []
            for r in result:
                if isinstance(r, Tensor) and needs_grad:
                    r.requires_grad = True
                    r._grad_fn = cls
                    r._ctx = ctx
                    r._prev = tuple(tensor_args)
                out.append(r)
            return tuple(out)

        return result


def _toposort(tensor) -> list:
    """Topological sort of the computation graph rooted at *tensor*."""
    order = []
    visited = set()

    def _visit(t):
        if id(t) in visited:
            return
        visited.add(id(t))
        for parent in t._prev:
            _visit(parent)
        order.append(t)

    _visit(tensor)
    return order


def backward(root, grad_output=None):
    """Run reverse-mode AD from *root* (must be scalar or provide grad_output)."""
    from cuDeep.tensor import Tensor, ones_like

    if grad_output is None:
        assert root._data.numel() == 1, (
            "backward() requires grad_output for non-scalar tensors"
        )
        grad_output = ones_like(root)

    root.grad = grad_output

    order = _toposort(root)
    for t in reversed(order):
        if t._grad_fn is None or t.grad is None:
            continue
        grads = t._grad_fn.backward(t._ctx, t.grad)
        if not isinstance(grads, tuple):
            grads = (grads,)
        for parent, g in zip(t._prev, grads):
            if g is None or not parent.requires_grad:
                continue
            if parent.grad is None:
                parent.grad = g
            else:
                parent.grad = Tensor._wrap(parent.grad._data + g._data)
