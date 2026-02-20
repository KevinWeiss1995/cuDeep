"""cuDeep: Ultra-high performance deep learning library implemented in CUDA."""

__version__ = "0.1.0"

from cuDeep.tensor import (
    Tensor,
    zeros_like, ones_like, randn_like,
    relu, sigmoid, tanh, gelu, silu, softmax,
    mse_loss, cross_entropy_loss, broadcast_add, conv2d,
)
from cuDeep._core import DType, Stream, Event, MemoryPool, device_info
from cuDeep.autograd import no_grad
from cuDeep import nn
from cuDeep import optim
from cuDeep import layers
from cuDeep import autograd
from cuDeep import init
from cuDeep import lr_scheduler

__all__ = [
    "Tensor", "DType", "Stream", "Event", "MemoryPool",
    "zeros_like", "ones_like", "randn_like",
    "relu", "sigmoid", "tanh", "gelu", "silu", "softmax",
    "mse_loss", "cross_entropy_loss", "broadcast_add", "conv2d",
    "device_info", "no_grad",
    "nn", "optim", "layers", "autograd", "init", "lr_scheduler",
]
