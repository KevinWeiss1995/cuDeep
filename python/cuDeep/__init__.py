"""cuDeep: Ultra-high performance deep learning library implemented in CUDA."""

__version__ = "0.1.0"

from cuDeep._core import Tensor, DType, Layout, Stream, Event, MemoryPool
from cuDeep import nn
from cuDeep import optim
from cuDeep import utils

__all__ = [
    "Tensor",
    "DType",
    "Layout",
    "Stream",
    "Event",
    "MemoryPool",
    "nn",
    "optim",
    "utils",
]
