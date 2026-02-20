"""Optimizers for cuDeep."""

from __future__ import annotations

from cuDeep._core import Tensor, sgd_update, adam_update, adamw_update


class Optimizer:
    """Base optimizer."""

    def __init__(self, params, lr=1e-3):
        self.params = list(params)
        self.lr = lr

    def step(self, grads):
        """Apply one optimization step. grads is a list of Tensor gradients
        matching self.params in order."""
        raise NotImplementedError

    def zero_grad(self, grads):
        """Zero out gradient tensors."""
        for g in grads:
            g.zero_()


class SGD(Optimizer):
    def __init__(self, params, lr=1e-2, momentum=0.0, weight_decay=0.0):
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._velocity = [None] * len(self.params)

    def step(self, grads):
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            vel = self._velocity[i]
            if self.momentum != 0.0 and vel is None:
                vel = Tensor.zeros(param.shape(), param.dtype())
                self._velocity[i] = vel
            sgd_update(param, grad,
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

    def step(self, grads):
        self._step_count += 1
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            if self._m[i] is None:
                self._m[i] = Tensor.zeros(param.shape(), param.dtype())
                self._v[i] = Tensor.zeros(param.shape(), param.dtype())
            adam_update(param, grad, self._m[i], self._v[i],
                        lr=self.lr,
                        beta1=self.betas[0],
                        beta2=self.betas[1],
                        eps=self.eps,
                        weight_decay=self.weight_decay,
                        step=self._step_count)


class AdamW(Optimizer):
    """Adam with decoupled weight decay (Loshchilov & Hutter 2019).

    Weight decay is applied directly to params (param -= lr * wd * param)
    before the Adam moment update, NOT mixed into the gradient like L2.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self._step_count = 0
        self._m = [None] * len(self.params)
        self._v = [None] * len(self.params)

    def step(self, grads):
        self._step_count += 1
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            if self._m[i] is None:
                self._m[i] = Tensor.zeros(param.shape(), param.dtype())
                self._v[i] = Tensor.zeros(param.shape(), param.dtype())
            adamw_update(param, grad, self._m[i], self._v[i],
                         lr=self.lr,
                         beta1=self.betas[0],
                         beta2=self.betas[1],
                         eps=self.eps,
                         weight_decay=self.weight_decay,
                         step=self._step_count)
