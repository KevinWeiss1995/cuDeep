"""Benchmark matmul against NumPy (CPU) baseline. GPU benchmarks require built extension."""

import time
import numpy as np


def bench_numpy_matmul(M, N, K, warmup=5, iters=50):
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)

    for _ in range(warmup):
        np.dot(A, B)

    start = time.perf_counter()
    for _ in range(iters):
        np.dot(A, B)
    elapsed = (time.perf_counter() - start) / iters

    gflops = (2.0 * M * N * K) / (elapsed * 1e9)
    print(f"NumPy matmul {M}x{K} @ {K}x{N}: {elapsed*1000:.3f} ms, {gflops:.1f} GFLOPS")


def bench_cudeep_matmul(M, N, K, warmup=10, iters=100):
    try:
        from cuDeep import Tensor
    except ImportError:
        print("cuDeep not built, skipping GPU benchmark")
        return

    A = Tensor.randn([M, K])
    B = Tensor.randn([K, N])

    for _ in range(warmup):
        A.matmul(B)

    start = time.perf_counter()
    for _ in range(iters):
        A.matmul(B)
    elapsed = (time.perf_counter() - start) / iters

    gflops = (2.0 * M * N * K) / (elapsed * 1e9)
    print(f"cuDeep matmul {M}x{K} @ {K}x{N}: {elapsed*1000:.3f} ms, {gflops:.1f} GFLOPS")


if __name__ == "__main__":
    for size in [128, 256, 512, 1024, 2048]:
        bench_numpy_matmul(size, size, size)
        bench_cudeep_matmul(size, size, size)
        print()
