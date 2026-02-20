"""Optimizers for cuDeep.

All optimizers consume ``Parameter.grad`` populated by ``loss.backward()``.
Call ``optimizer.zero_grad()`` before each forward pass to reset gradients.
"""

from __future__ import annotations
from cuDeep._core import (
    Tensor as _RawTensor,
    DType,
    sgd_update as _sgd_update,
    adam_update as _adam_update,
    adamw_update as _adamw_update,
)


class Optimizer:
    """Base optimizer."""

    def __init__(self, params, lr=1e-3):
        self.params = list(params)
        self.lr = lr

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=1e-2, momentum=0.0, weight_decay=0.0):
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._velocity = [None] * len(self.params)

    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            vel = self._velocity[i]
            if self.momentum != 0.0 and vel is None:
                vel = _RawTensor.zeros(list(param.shape()), param.dtype())
                self._velocity[i] = vel
            _sgd_update(param._data, param.grad._data,
                        velocity=vel,
                        lr=self.lr,
                        momentum=self.momentum,
                        weight_decay=self.weight_decay)


class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self._step_count = 0
        self._m = [None] * len(self.params)
        self._v = [None] * len(self.params)

    def step(self):
        self._step_count += 1
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            if self._m[i] is None:
                self._m[i] = _RawTensor.zeros(list(param.shape()), param.dtype())
                self._v[i] = _RawTensor.zeros(list(param.shape()), param.dtype())
            _adam_update(param._data, param.grad._data,
                         self._m[i], self._v[i],
                         lr=self.lr,
                         beta1=self.betas[0],
                         beta2=self.betas[1],
                         eps=self.eps,
                         weight_decay=self.weight_decay,
                         step=self._step_count)


class AdamW(Optimizer):
    """Adam with decoupled weight decay (Loshchilov & Hutter 2019)."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self._step_count = 0
        self._m = [None] * len(self.params)
        self._v = [None] * len(self.params)

    def step(self):
        self._step_count += 1
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            if self._m[i] is None:
                self._m[i] = _RawTensor.zeros(list(param.shape()), param.dtype())
                self._v[i] = _RawTensor.zeros(list(param.shape()), param.dtype())
            _adamw_update(param._data, param.grad._data,
                          self._m[i], self._v[i],
                          lr=self.lr,
                          beta1=self.betas[0],
                          beta2=self.betas[1],
                          eps=self.eps,
                          weight_decay=self.weight_decay,
                          step=self._step_count)
