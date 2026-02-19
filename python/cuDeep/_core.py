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
    )
except ImportError as e:
    raise ImportError(
        "cuDeep native extension (_cudeep_core) not found. "
        "Build the project first: pip install -e . or cmake --build build"
    ) from e

__all__ = ["Tensor", "DType", "Layout", "Stream", "Event", "MemoryPool"]
