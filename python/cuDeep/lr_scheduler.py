"""Learning rate schedulers."""

from __future__ import annotations
import math


class _Scheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self._step_count = 0

    def step(self):
        self._step_count += 1
        self.optimizer.lr = self.get_lr()

    def get_lr(self):
        raise NotImplementedError


class StepLR(_Scheduler):
    """Decay LR by *gamma* every *step_size* epochs."""

    def __init__(self, optimizer, step_size, gamma=0.1):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self):
        return self.base_lr * (self.gamma ** (self._step_count // self.step_size))


class ExponentialLR(_Scheduler):
    """Decay LR by *gamma* every step."""

    def __init__(self, optimizer, gamma):
        super().__init__(optimizer)
        self.gamma = gamma

    def get_lr(self):
        return self.base_lr * (self.gamma ** self._step_count)


class CosineAnnealingLR(_Scheduler):
    """Cosine annealing from *base_lr* to *eta_min* over *T_max* steps."""

    def __init__(self, optimizer, T_max, eta_min=0.0):
        super().__init__(optimizer)
        self.T_max = T_max
        self.eta_min = eta_min

    def get_lr(self):
        return self.eta_min + (self.base_lr - self.eta_min) * (
            1 + math.cos(math.pi * self._step_count / self.T_max)
        ) / 2


class LinearWarmupLR(_Scheduler):
    """Linear warmup for *warmup_steps*, then constant at *base_lr*."""

    def __init__(self, optimizer, warmup_steps):
        super().__init__(optimizer)
        self.warmup_steps = warmup_steps

    def get_lr(self):
        if self._step_count < self.warmup_steps:
            return self.base_lr * (self._step_count / self.warmup_steps)
        return self.base_lr


class OneCycleLR(_Scheduler):
    """1-cycle policy: warmup to max_lr, then cosine decay to min_lr."""

    def __init__(self, optimizer, max_lr, total_steps, pct_start=0.3, div_factor=25.0,
                 final_div_factor=1e4):
        super().__init__(optimizer)
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.init_lr = max_lr / div_factor
        self.min_lr = self.init_lr / final_div_factor

    def get_lr(self):
        t = self._step_count / max(self.total_steps, 1)
        if t < self.pct_start:
            frac = t / self.pct_start
            return self.init_lr + (self.max_lr - self.init_lr) * frac
        else:
            frac = (t - self.pct_start) / (1.0 - self.pct_start)
            return self.min_lr + (self.max_lr - self.min_lr) * (
                1 + math.cos(math.pi * frac)
            ) / 2
