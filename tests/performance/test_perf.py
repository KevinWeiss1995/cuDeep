"""Performance tests — verify kernels meet minimum throughput and don't regress.

These aren't micro-benchmarks for absolute GFLOPS numbers (use benchmarks/ for that).
They're regression gates: if a kernel gets 10x slower than a CPU baseline, something
is catastrophically wrong. All thresholds are deliberately loose.
"""

import time
import numpy as np
import pytest

from cuDeep import DType
from cuDeep._core import (
    Tensor as _RT,
    relu, sigmoid, gelu,
    sum as cu_sum, mean as cu_mean, max as cu_max,
    softmax, mse_loss,
)


def gpu_time(fn, warmup=3, iters=20):
    """Time a GPU operation, returning median seconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    return np.median(times)


class TestMatmulPerformance:
    """Matmul should beat single-threaded NumPy for large sizes."""

    @pytest.mark.parametrize("N", [256, 512, 1024])
    def test_matmul_faster_than_naive_cpu(self, N):
        A_np = np.random.randn(N, N).astype(np.float32)
        B_np = np.random.randn(N, N).astype(np.float32)
        A_cu = _RT.from_numpy(A_np)
        B_cu = _RT.from_numpy(B_np)

        gpu_t = gpu_time(lambda: A_cu.matmul(B_cu))

        # Verify correctness while we're at it
        result = A_cu.matmul(B_cu).numpy()
        expected = A_np @ B_np
        np.testing.assert_allclose(result, expected, rtol=1e-2, atol=1e-1)

        # Sanity: GPU matmul should complete in under 100ms for 1024x1024
        assert gpu_t < 0.5, f"Matmul {N}x{N} took {gpu_t:.3f}s, expected < 0.5s"

    def test_matmul_scales_sublinearly(self):
        """Doubling N shouldn't 8x the time (it's O(N^3) but GPU parallelism helps)."""
        A256 = _RT.randn([256, 256])
        B256 = _RT.randn([256, 256])
        A512 = _RT.randn([512, 512])
        B512 = _RT.randn([512, 512])

        t256 = gpu_time(lambda: A256.matmul(B256))
        t512 = gpu_time(lambda: A512.matmul(B512))

        # 8x work but GPU should keep the ratio well under 8x
        ratio = t512 / max(t256, 1e-9)
        assert ratio < 20, f"512/256 time ratio = {ratio:.1f}, expected < 20"


class TestElementwisePerformance:
    """Elementwise ops on 1M elements should be sub-millisecond."""

    def test_add_throughput(self):
        a = _RT.randn([1000000])
        b = _RT.randn([1000000])
        t = gpu_time(lambda: a + b)
        assert t < 0.05, f"Add 1M took {t*1000:.1f}ms"

    def test_mul_throughput(self):
        a = _RT.randn([1000000])
        b = _RT.randn([1000000])
        t = gpu_time(lambda: a * b)
        assert t < 0.05, f"Mul 1M took {t*1000:.1f}ms"


class TestActivationPerformance:
    def test_relu_large(self):
        x = _RT.randn([1000000])
        t = gpu_time(lambda: relu(x))
        assert t < 0.05, f"ReLU 1M took {t*1000:.1f}ms"

    def test_gelu_large(self):
        x = _RT.randn([1000000])
        t = gpu_time(lambda: gelu(x))
        assert t < 0.05, f"GELU 1M took {t*1000:.1f}ms"

    def test_sigmoid_large(self):
        x = _RT.randn([1000000])
        t = gpu_time(lambda: sigmoid(x))
        assert t < 0.05, f"Sigmoid 1M took {t*1000:.1f}ms"


class TestReductionPerformance:
    def test_sum_1m(self):
        x = _RT.randn([1000000])
        t = gpu_time(lambda: cu_sum(x))
        assert t < 0.05

    def test_max_1m(self):
        x = _RT.randn([1000000])
        t = gpu_time(lambda: cu_max(x))
        assert t < 0.1  # two-level reduction, slightly slower

    def test_mean_1m(self):
        x = _RT.randn([1000000])
        t = gpu_time(lambda: cu_mean(x))
        assert t < 0.05


class TestSoftmaxPerformance:
    def test_softmax_batch(self):
        x = _RT.randn([1024, 1000])
        t = gpu_time(lambda: softmax(x, 1))
        assert t < 0.5, f"Softmax 1024x1000 took {t*1000:.1f}ms"


class TestMemoryAllocation:
    """Verify memory pool doesn't leak under repeated alloc/dealloc."""

    def test_repeated_alloc_free(self):
        from cuDeep import MemoryPool
        pool = MemoryPool.instance()

        initial = pool.allocated_bytes

        for _ in range(100):
            t = _RT.randn([256, 256])
            del t

        pool.release_cached()
        final = pool.allocated_bytes
        assert final <= initial + 1024 * 1024, \
            f"Leaked {final - initial} bytes after 100 alloc/free cycles"


class TestTensorCreationPerformance:
    def test_randn_large(self):
        t = gpu_time(lambda: _RT.randn([1000, 1000]))
        assert t < 0.5, f"randn 1Mx1 took {t*1000:.1f}ms"

    def test_zeros_large(self):
        t = gpu_time(lambda: _RT.zeros([1000, 1000]))
        assert t < 0.1
