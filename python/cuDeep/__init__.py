"""cuDeep: Ultra-high performance deep learning library implemented in CUDA."""

__version__ = "0.1.0"

from cuDeep._core import (
    Tensor, DType, Layout, Stream, Event, MemoryPool,
    # Functional ops
    relu, sigmoid, tanh_act, gelu, silu, leaky_relu,
    sum, mean, max, min,
    softmax, mse_loss, cross_entropy_loss,
    conv2d_forward, max_pool2d, avg_pool2d,
    batchnorm_forward, layernorm_forward,
    sgd_update, adam_update, adamw_update,
    broadcast_add, scalar_mul, device_info,
)
from cuDeep import nn
from cuDeep import optim
from cuDeep import layers
from cuDeep import utils

__all__ = [
    "Tensor", "DType", "Layout", "Stream", "Event", "MemoryPool",
    "relu", "sigmoid", "tanh_act", "gelu", "silu", "leaky_relu",
    "sum", "mean", "max", "min",
    "softmax", "mse_loss", "cross_entropy_loss",
    "conv2d_forward", "max_pool2d", "avg_pool2d",
    "batchnorm_forward", "layernorm_forward",
    "sgd_update", "adam_update", "adamw_update",
    "broadcast_add", "scalar_mul", "device_info",
    "nn", "optim", "layers", "utils",
]
