"""Optimizers for cuDeep."""

from __future__ import annotations

from cuDeep._core import Tensor


class Optimizer:
    """Base optimizer."""

    def __init__(self, params, lr=1e-3):
        self.params = params
        self.lr = lr

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        # TODO: zero out gradient tensors
        pass


class SGD(Optimizer):
    def __init__(self, params, lr=1e-2, momentum=0.0, weight_decay=0.0):
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._velocity = [None] * len(params)

    def step(self):
        # TODO: implement SGD update kernel
        raise NotImplementedError


class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self._step_count = 0
        self._m = [None] * len(params)
        self._v = [None] * len(params)

    def step(self):
        # TODO: implement Adam update kernel
        raise NotImplementedError


class AdamW(Optimizer):
    """Adam with decoupled weight decay."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self._step_count = 0

    def step(self):
        raise NotImplementedError
