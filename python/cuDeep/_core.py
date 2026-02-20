"""Bridge module that imports from the compiled C extension."""

try:
    from _cudeep_core import (  # type: ignore[import-not-found]
        Tensor,
        DType,
        Layout,
        Stream,
        Event,
        MemoryPool,
        # Elementwise
        relu, sigmoid, tanh_act, gelu, silu, leaky_relu,
        broadcast_add, scalar_mul, div_op,
        # Unary math
        neg, exp_op, log_op, sqrt_op, pow_op, abs_op, clamp_op, gt_mask,
        # Reductions
        sum, mean, max, min, sum_reduce_rows,
        # Softmax / loss
        softmax, mse_loss, cross_entropy_loss,
        # Conv / pool / norm
        conv2d_forward, conv2d_backward_data, conv2d_backward_weight,
        max_pool2d, avg_pool2d,
        batchnorm_forward, layernorm_forward,
        # Backward helpers
        activation_backward,
        # Optimizers
        sgd_update, adam_update, adamw_update,
        # Utility
        device_info,
    )
except ImportError as e:
    raise ImportError(
        "cuDeep native extension (_cudeep_core) not found. "
        "Build the project first: pip install -e . or cmake --build build"
    ) from e
