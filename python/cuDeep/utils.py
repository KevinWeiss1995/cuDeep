"""Utility functions for cuDeep."""

from __future__ import annotations

from cuDeep._core import Tensor, MemoryPool, device_info as _device_info


def memory_stats():
    """Return current GPU memory pool stats."""
    pool = MemoryPool.instance()
    return {
        "allocated_bytes": pool.allocated_bytes,
        "cached_bytes": pool.cached_bytes,
    }


def release_memory():
    """Release all cached memory back to CUDA."""
    MemoryPool.instance().release_cached()


def device_info():
    """Query CUDA device properties."""
    return _device_info()
