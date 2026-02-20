"""MatMul benchmark: cuDeep GPU vs NumPy CPU.

Uses CUDA synchronization for accurate GPU timing and reports GFLOPS.
"""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "python"))
sys.path.insert(0, str(_root / "build"))

import time
import numpy as np


def sync_gpu():
    """Force all pending GPU work to complete."""
    from cuDeep._core import Tensor as _RT, DType
    _RT.zeros([1], DType.float32).numpy()


def bench_numpy(M, N, K, warmup=5, iters=50):
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    for _ in range(warmup):
        np.dot(A, B)
    t0 = time.perf_counter()
    for _ in range(iters):
        np.dot(A, B)
    elapsed = (time.perf_counter() - t0) / iters
    gflops = (2.0 * M * N * K) / (elapsed * 1e9)
    return elapsed, gflops


def bench_cudeep(M, N, K, warmup=10, iters=100):
    from cuDeep._core import Tensor as _RT, DType
    A = _RT.randn([M, K], DType.float32)
    B = _RT.randn([K, N], DType.float32)
    for _ in range(warmup):
        A.matmul(B)
    sync_gpu()
    t0 = time.perf_counter()
    for _ in range(iters):
        A.matmul(B)
    sync_gpu()
    elapsed = (time.perf_counter() - t0) / iters
    gflops = (2.0 * M * N * K) / (elapsed * 1e9)
    return elapsed, gflops


if __name__ == "__main__":
    from cuDeep._core import device_info
    info = device_info()
    print(f"Device: {info['name']}")
    print(f"{'Size':>8s}  {'NumPy ms':>10s}  {'NumPy GF':>10s}  {'cuDeep ms':>10s}  {'cuDeep GF':>10s}  {'Speedup':>8s}")
    print("-" * 72)

    for size in [128, 256, 512, 1024, 2048]:
        np_time, np_gf = bench_numpy(size, size, size)
        cu_time, cu_gf = bench_cudeep(size, size, size)
        speedup = np_time / cu_time
        print(f"{size:>8d}  {np_time*1000:>10.3f}  {np_gf:>10.1f}  {cu_time*1000:>10.3f}  {cu_gf:>10.1f}  {speedup:>7.2f}x")
