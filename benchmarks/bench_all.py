"""Comprehensive cuDeep kernel benchmark suite.

Benchmarks every kernel category against NumPy/CPU baselines with
accurate GPU-synchronized timing. Reports latency, throughput, and
speedup in a formatted table.
"""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "python"))
sys.path.insert(0, str(_root / "build"))

import time
import numpy as np
from dataclasses import dataclass

from cuDeep._core import (
    Tensor as _RT, DType, device_info,
    relu, sigmoid, gelu, silu, tanh_act,
    softmax, mse_loss, cross_entropy_loss,
    sum as cu_sum, mean as cu_mean, max as cu_max, min as cu_min,
    conv2d_forward,
    batchnorm_forward, layernorm_forward,
    neg, exp_op, log_op, sqrt_op,
)


def sync():
    _RT.zeros([1], DType.float32).numpy()


@dataclass
class Result:
    name: str
    desc: str
    gpu_ms: float
    cpu_ms: float

    @property
    def speedup(self):
        return self.cpu_ms / self.gpu_ms if self.gpu_ms > 0 else float("inf")


def gpu_bench(fn, warmup=10, iters=100):
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - t0) / iters * 1000


def cpu_bench(fn, warmup=3, iters=30):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1000


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def bench_matmul():
    results = []
    for N in [256, 512, 1024, 2048]:
        a_np = np.random.randn(N, N).astype(np.float32)
        b_np = np.random.randn(N, N).astype(np.float32)
        a = _RT.from_numpy(a_np)
        b = _RT.from_numpy(b_np)
        gms = gpu_bench(lambda: a.matmul(b))
        cms = cpu_bench(lambda: a_np @ b_np)
        flops = 2.0 * N * N * N
        results.append(Result(
            f"MatMul {N}x{N}",
            f"{flops/gms/1e6:.0f} / {flops/cms/1e6:.0f} GFLOPS",
            gms, cms))
    return results


def bench_elementwise():
    results = []
    for n in [100_000, 1_000_000, 10_000_000]:
        a_np = np.random.randn(n).astype(np.float32)
        b_np = np.random.randn(n).astype(np.float32)
        a = _RT.from_numpy(a_np)
        b = _RT.from_numpy(b_np)
        tag = f"{n//1000}K" if n < 1_000_000 else f"{n//1_000_000}M"

        gms = gpu_bench(lambda: a + b)
        cms = cpu_bench(lambda: a_np + b_np)
        bw = n * 4 * 3 / gms / 1e6
        results.append(Result(f"Add {tag}", f"{bw:.1f} GB/s", gms, cms))

        gms = gpu_bench(lambda: a * b)
        cms = cpu_bench(lambda: a_np * b_np)
        results.append(Result(f"Mul {tag}", "", gms, cms))
    return results


def bench_reductions():
    results = []
    for n in [100_000, 1_000_000, 10_000_000]:
        x_np = np.random.randn(n).astype(np.float32)
        x = _RT.from_numpy(x_np)
        tag = f"{n//1000}K" if n < 1_000_000 else f"{n//1_000_000}M"

        gms = gpu_bench(lambda: cu_sum(x))
        cms = cpu_bench(lambda: np.sum(x_np))
        results.append(Result(f"Sum {tag}", "", gms, cms))

        gms = gpu_bench(lambda: cu_max(x))
        cms = cpu_bench(lambda: np.max(x_np))
        results.append(Result(f"Max {tag}", "", gms, cms))
    return results


def bench_activations():
    results = []
    n = 1_000_000
    x_np = np.random.randn(n).astype(np.float32)
    x = _RT.from_numpy(x_np)

    for name, gpu_fn, cpu_fn in [
        ("ReLU",    lambda: relu(x),     lambda: np.maximum(x_np, 0)),
        ("Sigmoid", lambda: sigmoid(x),  lambda: 1/(1+np.exp(-x_np))),
        ("GELU",    lambda: gelu(x),     lambda: x_np * 0.5 * (1 + np.tanh(0.7978845608 * (x_np + 0.044715 * x_np**3)))),
        ("SiLU",    lambda: silu(x),     lambda: x_np / (1 + np.exp(-x_np))),
    ]:
        gms = gpu_bench(gpu_fn)
        cms = cpu_bench(cpu_fn)
        results.append(Result(f"{name} 1M", "", gms, cms))
    return results


def bench_unary_math():
    results = []
    n = 1_000_000
    x_np = np.abs(np.random.randn(n).astype(np.float32)) + 0.01
    x = _RT.from_numpy(x_np)

    for name, gpu_fn, cpu_fn in [
        ("Neg",  lambda: neg(x),     lambda: -x_np),
        ("Exp",  lambda: exp_op(x),  lambda: np.exp(x_np)),
        ("Log",  lambda: log_op(x),  lambda: np.log(x_np)),
        ("Sqrt", lambda: sqrt_op(x), lambda: np.sqrt(x_np)),
    ]:
        gms = gpu_bench(gpu_fn)
        cms = cpu_bench(cpu_fn)
        results.append(Result(f"{name} 1M", "", gms, cms))
    return results


def bench_softmax():
    results = []
    for batch, dim in [(64, 128), (128, 512), (128, 4096), (32, 32768)]:
        x_np = np.random.randn(batch, dim).astype(np.float32)
        x = _RT.from_numpy(x_np)
        from scipy.special import softmax as sp_softmax

        gms = gpu_bench(lambda: softmax(x))
        cms = cpu_bench(lambda: sp_softmax(x_np, axis=1))
        results.append(Result(f"Softmax {batch}x{dim}", "", gms, cms))
    return results


def bench_loss():
    results = []
    for n in [1000, 10000, 100000]:
        p_np = np.random.randn(n).astype(np.float32)
        t_np = np.random.randn(n).astype(np.float32)
        p = _RT.from_numpy(p_np)
        t = _RT.from_numpy(t_np)
        tag = f"{n//1000}K" if n >= 1000 else str(n)

        gms = gpu_bench(lambda: mse_loss(p, t))
        cms = cpu_bench(lambda: np.mean((p_np - t_np)**2))
        results.append(Result(f"MSE {tag}", "", gms, cms))

    batch, C = 128, 100
    logits_np = np.random.randn(batch, C).astype(np.float32)
    logits = _RT.from_numpy(logits_np)
    targets = np.random.randint(0, C, batch).astype(np.int32)
    targets_rt = _RT.from_numpy(targets.astype(np.float32))

    gms = gpu_bench(lambda: cross_entropy_loss(logits, targets))

    def np_ce():
        m = logits_np - logits_np.max(axis=1, keepdims=True)
        e = np.exp(m)
        s = e / e.sum(axis=1, keepdims=True)
        return -np.mean(np.log(s[np.arange(batch), targets] + 1e-9))

    cms = cpu_bench(np_ce)
    results.append(Result(f"CrossEntropy {batch}x{C}", "", gms, cms))
    return results


def bench_conv2d():
    results = []
    configs = [
        (1,  32, 64, 32, 32, 3, 1, 1),
        (8,  64, 128, 16, 16, 3, 1, 1),
        (16, 128, 256, 8, 8, 3, 1, 1),
    ]
    for B, IC, OC, H, W, K, S, P in configs:
        x_np = np.random.randn(B, IC, H, W).astype(np.float32)
        w_np = np.random.randn(OC, IC, K, K).astype(np.float32)
        x = _RT.from_numpy(x_np)
        w = _RT.from_numpy(w_np)

        gms = gpu_bench(lambda: conv2d_forward(x, w, None, [S, S], [P, P]))
        flops = 2.0 * B * OC * H * W * IC * K * K
        results.append(Result(
            f"Conv2d {B}x{IC}x{H}x{W} k{K}",
            f"{flops/gms/1e6:.0f} GFLOPS",
            gms, 0))
    return results


def bench_norms():
    results = []

    x_np = np.random.randn(32, 64, 8, 8).astype(np.float32)
    x = _RT.from_numpy(x_np)
    w = _RT.ones([64], DType.float32)
    b = _RT.zeros([64], DType.float32)
    rm = _RT.zeros([64], DType.float32)
    rv = _RT.ones([64], DType.float32)
    gms = gpu_bench(lambda: batchnorm_forward(x, w, b, rm, rv, 1e-5, 0.1, True))
    results.append(Result("BatchNorm2d [32,64,8,8]", "", gms, 0))

    for dim in [256, 512, 2048]:
        x_np = np.random.randn(128, dim).astype(np.float32)
        x = _RT.from_numpy(x_np)
        wl = _RT.ones([dim], DType.float32)
        bl = _RT.zeros([dim], DType.float32)
        gms = gpu_bench(lambda: layernorm_forward(x, wl, bl, dim, 1e-5))
        results.append(Result(f"LayerNorm [128,{dim}]", "", gms, 0))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_section(title, results):
    print(f"\n{'='*74}")
    print(f"  {title}")
    print(f"{'='*74}")
    print(f"  {'Kernel':<30s}  {'GPU ms':>8s}  {'CPU ms':>8s}  {'Speedup':>8s}  {'Notes'}")
    print(f"  {'-'*30}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*20}")
    for r in results:
        cpu_str = f"{r.cpu_ms:8.3f}" if r.cpu_ms > 0 else "     N/A"
        sp_str = f"{r.speedup:7.1f}x" if r.cpu_ms > 0 else "     N/A"
        print(f"  {r.name:<30s}  {r.gpu_ms:8.3f}  {cpu_str}  {sp_str}  {r.desc}")


def main():
    info = device_info()
    print(f"cuDeep Kernel Benchmark Suite")
    print(f"Device: {info['name']}")
    print(f"NumPy: {np.__version__} (CPU baseline)")
    print(f"dtype: float32")

    try:
        import scipy
        has_scipy = True
    except ImportError:
        has_scipy = False

    print_section("Matrix Multiplication", bench_matmul())
    print_section("Elementwise Operations", bench_elementwise())
    print_section("Reductions", bench_reductions())
    print_section("Activations", bench_activations())
    print_section("Unary Math", bench_unary_math())

    if has_scipy:
        print_section("Softmax", bench_softmax())
    else:
        print("\n[Skipping Softmax benchmark — install scipy for CPU baseline]")

    print_section("Loss Functions", bench_loss())
    print_section("Convolution", bench_conv2d())
    print_section("Normalization", bench_norms())

    print(f"\n{'='*74}")
    print("  Benchmark complete.")
    print(f"{'='*74}")


if __name__ == "__main__":
    main()
