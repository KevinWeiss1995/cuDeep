"""
Bridge module that imports from the compiled C extension.

Allows graceful error messages if the native module isn't built yet.
"""

try:
    from _cudeep_core import (  # type: ignore[import-not-found]
        Tensor,
        DType,
        Layout,
        Stream,
        Event,
        MemoryPool,
        # Functional: activations
        relu,
        sigmoid,
        tanh_act,
        gelu,
        silu,
        leaky_relu,
        # Functional: reductions
        sum,
        mean,
        max,
        min,
        # Functional: softmax / loss
        softmax,
        mse_loss,
        cross_entropy_loss,
        # Functional: conv / pool / norm
        conv2d_forward,
        max_pool2d,
        avg_pool2d,
        batchnorm_forward,
        layernorm_forward,
        # Functional: optimizer steps
        sgd_update,
        adam_update,
        adamw_update,
        # Utility
        broadcast_add,
        scalar_mul,
        device_info,
    )
except ImportError as e:
    raise ImportError(
        "cuDeep native extension (_cudeep_core) not found. "
        "Build the project first: pip install -e . or cmake --build build"
    ) from e

__all__ = [
    "Tensor", "DType", "Layout", "Stream", "Event", "MemoryPool",
    "relu", "sigmoid", "tanh_act", "gelu", "silu", "leaky_relu",
    "sum", "mean", "max", "min",
    "softmax", "mse_loss", "cross_entropy_loss",
    "conv2d_forward", "max_pool2d", "avg_pool2d",
    "batchnorm_forward", "layernorm_forward",
    "sgd_update", "adam_update", "adamw_update",
    "broadcast_add", "scalar_mul", "device_info",
]
