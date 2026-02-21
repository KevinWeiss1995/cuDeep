"""Comprehensive cuDeep kernel benchmark suite.

Benchmarks every exposed operation against NumPy/CPU baselines with
accurate GPU-synchronized timing.  Reports latency, throughput, and
speedup in a formatted table.

Coverage: matmul, elementwise (add/sub/mul/div/scalar_mul/broadcast_add),
reductions (sum/mean/max/min/sum_reduce_rows), activations (relu/sigmoid/
tanh/gelu/silu/leaky_relu + backward), unary math (neg/exp/log/sqrt/pow/
abs/clamp/gt_mask), softmax, loss (mse/cross-entropy), conv2d (fwd/bwd
data/bwd weight), pooling (maxpool/avgpool), normalization (batchnorm/
layernorm), optimizer steps (sgd/adam/adamw), tensor ops (transpose/
contiguous/fill/zero).
"""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "python"))
sys.path.insert(0, str(_root / "build"))

import time
import math
import numpy as np
from dataclasses import dataclass, field

from cuDeep._core import (
    Tensor as _RT, DType, device_info,
    # activations
    relu, sigmoid, gelu, silu, tanh_act, leaky_relu,
    # reductions
    sum as cu_sum, mean as cu_mean, max as cu_max, min as cu_min,
    sum_reduce_rows,
    # softmax / loss
    softmax, mse_loss, cross_entropy_loss,
    # conv
    conv2d_forward, conv2d_backward_data, conv2d_backward_weight,
    # pooling
    max_pool2d, avg_pool2d,
    # norm
    batchnorm_forward, layernorm_forward,
    # optimizer steps
    sgd_update, adam_update, adamw_update,
    # elementwise extras
    scalar_mul, broadcast_add, div_op,
    # unary
    neg, exp_op, log_op, sqrt_op, pow_op, abs_op, clamp_op, gt_mask,
    # activation backward
    activation_backward,
)


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def sync():
    """Force GPU synchronization via a tiny round-trip."""
    _RT.zeros([1], DType.float32).numpy()


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


@dataclass
class Result:
    name: str
    desc: str
    gpu_ms: float
    cpu_ms: float

    @property
    def speedup(self):
        return self.cpu_ms / self.gpu_ms if self.gpu_ms > 0 else float("inf")


# ---------------------------------------------------------------------------
# 1. Matrix Multiplication
# ---------------------------------------------------------------------------

def bench_matmul():
    results = []
    configs = [
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (128, 4096, 512),   # tall-skinny
        (4096, 128, 512),   # wide-short
    ]
    for M, K, N in configs:
        a_np = np.random.randn(M, K).astype(np.float32)
        b_np = np.random.randn(K, N).astype(np.float32)
        a = _RT.from_numpy(a_np)
        b = _RT.from_numpy(b_np)
        gms = gpu_bench(lambda: a.matmul(b))
        cms = cpu_bench(lambda: a_np @ b_np)
        flops = 2.0 * M * K * N
        results.append(Result(
            f"MatMul {M}x{K}x{N}",
            f"{flops/gms/1e6:.0f} / {flops/cms/1e6:.0f} GFLOPS",
            gms, cms))
    return results


# ---------------------------------------------------------------------------
# 2. Elementwise Operations
# ---------------------------------------------------------------------------

def bench_elementwise():
    results = []
    sizes = [100_000, 1_000_000, 10_000_000]

    for n in sizes:
        a_np = np.random.randn(n).astype(np.float32)
        b_np = np.random.randn(n).astype(np.float32) + 0.1
        a = _RT.from_numpy(a_np)
        b = _RT.from_numpy(b_np)
        tag = f"{n // 1000}K" if n < 1_000_000 else f"{n // 1_000_000}M"
        bytes_3 = n * 4 * 3  # 2 reads + 1 write

        gms = gpu_bench(lambda: a + b)
        cms = cpu_bench(lambda: a_np + b_np)
        results.append(Result(f"Add {tag}", f"{bytes_3/gms/1e6:.1f} GB/s", gms, cms))

        gms = gpu_bench(lambda: a - b)
        cms = cpu_bench(lambda: a_np - b_np)
        results.append(Result(f"Sub {tag}", f"{bytes_3/gms/1e6:.1f} GB/s", gms, cms))

        gms = gpu_bench(lambda: a * b)
        cms = cpu_bench(lambda: a_np * b_np)
        results.append(Result(f"Mul {tag}", f"{bytes_3/gms/1e6:.1f} GB/s", gms, cms))

        gms = gpu_bench(lambda: div_op(a, b))
        cms = cpu_bench(lambda: a_np / b_np)
        results.append(Result(f"Div {tag}", f"{bytes_3/gms/1e6:.1f} GB/s", gms, cms))

    n = 10_000_000
    a_np = np.random.randn(n).astype(np.float32)
    a = _RT.from_numpy(a_np)
    gms = gpu_bench(lambda: scalar_mul(a, 2.5))
    cms = cpu_bench(lambda: a_np * 2.5)
    results.append(Result("ScalarMul 10M", "", gms, cms))

    rows, cols = 4096, 1024
    m_np = np.random.randn(rows, cols).astype(np.float32)
    r_np = np.random.randn(cols).astype(np.float32)
    m_rt = _RT.from_numpy(m_np)
    r_rt = _RT.from_numpy(r_np)
    gms = gpu_bench(lambda: broadcast_add(m_rt, r_rt))
    cms = cpu_bench(lambda: m_np + r_np)
    results.append(Result(f"BcastAdd {rows}x{cols}", "", gms, cms))

    return results


# ---------------------------------------------------------------------------
# 3. Reductions
# ---------------------------------------------------------------------------

def bench_reductions():
    results = []
    sizes = [100_000, 1_000_000, 10_000_000]

    for n in sizes:
        x_np = np.random.randn(n).astype(np.float32)
        x = _RT.from_numpy(x_np)
        tag = f"{n // 1000}K" if n < 1_000_000 else f"{n // 1_000_000}M"

        gms = gpu_bench(lambda: cu_sum(x))
        cms = cpu_bench(lambda: np.sum(x_np))
        results.append(Result(f"Sum {tag}", "", gms, cms))

        gms = gpu_bench(lambda: cu_mean(x))
        cms = cpu_bench(lambda: np.mean(x_np))
        results.append(Result(f"Mean {tag}", "", gms, cms))

        gms = gpu_bench(lambda: cu_max(x))
        cms = cpu_bench(lambda: np.max(x_np))
        results.append(Result(f"Max {tag}", "", gms, cms))

        gms = gpu_bench(lambda: cu_min(x))
        cms = cpu_bench(lambda: np.min(x_np))
        results.append(Result(f"Min {tag}", "", gms, cms))

    rows, cols = 2048, 512
    x_np = np.random.randn(rows, cols).astype(np.float32)
    x = _RT.from_numpy(x_np)
    gms = gpu_bench(lambda: sum_reduce_rows(x))
    cms = cpu_bench(lambda: np.sum(x_np, axis=0))
    results.append(Result(f"SumRows {rows}x{cols}", "", gms, cms))

    return results


# ---------------------------------------------------------------------------
# 4. Activations (forward)
# ---------------------------------------------------------------------------

def bench_activations():
    results = []
    n = 4_000_000
    x_np = np.random.randn(n).astype(np.float32)
    x = _RT.from_numpy(x_np)

    acts = [
        ("ReLU",      lambda: relu(x),              lambda: np.maximum(x_np, 0)),
        ("Sigmoid",   lambda: sigmoid(x),           lambda: 1.0 / (1.0 + np.exp(-x_np))),
        ("Tanh",      lambda: tanh_act(x),          lambda: np.tanh(x_np)),
        ("GELU",      lambda: gelu(x),              lambda: x_np * 0.5 * (1 + np.tanh(0.7978845608 * (x_np + 0.044715 * x_np**3)))),
        ("SiLU",      lambda: silu(x),              lambda: x_np / (1 + np.exp(-x_np))),
        ("LeakyReLU", lambda: leaky_relu(x, 0.01),  lambda: np.where(x_np > 0, x_np, 0.01 * x_np)),
    ]
    tag = f"{n // 1_000_000}M"
    for name, gpu_fn, cpu_fn in acts:
        gms = gpu_bench(gpu_fn)
        cms = cpu_bench(cpu_fn)
        bw = n * 4 * 2 / gms / 1e6   # 1 read + 1 write
        results.append(Result(f"{name} {tag}", f"{bw:.1f} GB/s", gms, cms))
    return results


# ---------------------------------------------------------------------------
# 5. Activation backward
# ---------------------------------------------------------------------------

def bench_activation_backward():
    results = []
    n = 4_000_000
    x_np = np.random.randn(n).astype(np.float32)
    g_np = np.random.randn(n).astype(np.float32)
    x = _RT.from_numpy(x_np)
    g = _RT.from_numpy(g_np)
    tag = f"{n // 1_000_000}M"

    for name in ["relu", "sigmoid", "tanh", "gelu", "silu", "leaky_relu"]:
        gms = gpu_bench(lambda: activation_backward(g, x, name, 0.01))
        results.append(Result(f"Bwd {name} {tag}", "", gms, 0))
    return results


# ---------------------------------------------------------------------------
# 6. Unary Math
# ---------------------------------------------------------------------------

def bench_unary_math():
    results = []
    n = 4_000_000
    x_np = np.abs(np.random.randn(n).astype(np.float32)) + 0.01
    x = _RT.from_numpy(x_np)
    tag = f"{n // 1_000_000}M"

    ops = [
        ("Neg",   lambda: neg(x),              lambda: -x_np),
        ("Exp",   lambda: exp_op(x),           lambda: np.exp(x_np)),
        ("Log",   lambda: log_op(x),           lambda: np.log(x_np)),
        ("Sqrt",  lambda: sqrt_op(x),          lambda: np.sqrt(x_np)),
        ("Pow2",  lambda: pow_op(x, 2.0),      lambda: x_np ** 2),
        ("Pow0.5",lambda: pow_op(x, 0.5),      lambda: x_np ** 0.5),
        ("Abs",   lambda: abs_op(x),           lambda: np.abs(x_np)),
        ("Clamp", lambda: clamp_op(x, 0.1, 2.0), lambda: np.clip(x_np, 0.1, 2.0)),
        ("GtMask",lambda: gt_mask(x, 0.5),     lambda: (x_np > 0.5).astype(np.float32)),
    ]
    for name, gpu_fn, cpu_fn in ops:
        gms = gpu_bench(gpu_fn)
        cms = cpu_bench(cpu_fn)
        results.append(Result(f"{name} {tag}", "", gms, cms))
    return results


# ---------------------------------------------------------------------------
# 7. Softmax
# ---------------------------------------------------------------------------

def _np_softmax(x, axis=1):
    m = x - x.max(axis=axis, keepdims=True)
    e = np.exp(m)
    return e / e.sum(axis=axis, keepdims=True)


def bench_softmax():
    results = []
    for batch, dim in [(64, 128), (128, 512), (128, 4096), (32, 32768)]:
        x_np = np.random.randn(batch, dim).astype(np.float32)
        x = _RT.from_numpy(x_np)
        gms = gpu_bench(lambda: softmax(x))
        cms = cpu_bench(lambda: _np_softmax(x_np))
        results.append(Result(f"Softmax {batch}x{dim}", "", gms, cms))
    return results


# ---------------------------------------------------------------------------
# 8. Loss Functions
# ---------------------------------------------------------------------------

def bench_loss():
    results = []
    for n in [1_000, 10_000, 100_000, 1_000_000]:
        p_np = np.random.randn(n).astype(np.float32)
        t_np = np.random.randn(n).astype(np.float32)
        p = _RT.from_numpy(p_np)
        t = _RT.from_numpy(t_np)
        tag = f"{n // 1000}K" if n >= 1000 else str(n)
        gms = gpu_bench(lambda: mse_loss(p, t))
        cms = cpu_bench(lambda: np.mean((p_np - t_np) ** 2))
        results.append(Result(f"MSE {tag}", "", gms, cms))

    for batch, C in [(64, 100), (128, 1000), (256, 10000)]:
        logits_np = np.random.randn(batch, C).astype(np.float32)
        logits = _RT.from_numpy(logits_np)
        targets = np.random.randint(0, C, batch).astype(np.int32)

        gms = gpu_bench(lambda: cross_entropy_loss(logits, targets))

        def _np_ce(l=logits_np, tgt=targets, b=batch):
            m = l - l.max(axis=1, keepdims=True)
            e = np.exp(m)
            s = e / e.sum(axis=1, keepdims=True)
            return -np.mean(np.log(s[np.arange(b), tgt] + 1e-9))

        cms = cpu_bench(_np_ce)
        results.append(Result(f"CE {batch}x{C}", "", gms, cms))

    return results


# ---------------------------------------------------------------------------
# 9. Conv2d (forward + backward)
# ---------------------------------------------------------------------------

def bench_conv2d():
    results = []
    configs = [
        (1,  32, 64,  32, 32, 3, 1, 1),
        (8,  64, 128, 16, 16, 3, 1, 1),
        (16, 128, 256, 8,  8,  3, 1, 1),
        (4,  3,   64, 64, 64, 3, 1, 1),  # first-layer-like
    ]
    for B, IC, OC, H, W, K, S, P in configs:
        OH = (H + 2 * P - K) // S + 1
        OW = (W + 2 * P - K) // S + 1
        x_np = np.random.randn(B, IC, H, W).astype(np.float32)
        w_np = np.random.randn(OC, IC, K, K).astype(np.float32)
        x = _RT.from_numpy(x_np)
        w = _RT.from_numpy(w_np)

        gms = gpu_bench(lambda: conv2d_forward(x, w, None, [S, S], [P, P]))
        flops = 2.0 * B * OC * OH * OW * IC * K * K
        results.append(Result(
            f"Fwd {B}x{IC}x{H}x{W} k{K}",
            f"{flops/gms/1e6:.0f} GFLOPS", gms, 0))

        go_np = np.random.randn(B, OC, OH, OW).astype(np.float32)
        go = _RT.from_numpy(go_np)
        ishape = [B, IC, H, W]
        wshape = [OC, IC, K, K]

        gms_bd = gpu_bench(lambda: conv2d_backward_data(go, w, ishape, [S, S], [P, P]))
        results.append(Result(
            f"BwdData {B}x{IC}x{H}x{W}",
            f"{flops/gms_bd/1e6:.0f} GFLOPS", gms_bd, 0))

        gms_bw = gpu_bench(lambda: conv2d_backward_weight(go, x, wshape, [S, S], [P, P]))
        results.append(Result(
            f"BwdWt {OC}x{IC}x{K}x{K}",
            f"{flops/gms_bw/1e6:.0f} GFLOPS", gms_bw, 0))

    return results


# ---------------------------------------------------------------------------
# 10. Pooling
# ---------------------------------------------------------------------------

def bench_pooling():
    results = []
    configs = [
        (8,  64,  32, 32, 2, 2),
        (16, 128, 16, 16, 2, 2),
        (32, 256, 8,  8,  2, 2),
    ]
    for B, C, H, W, K, S in configs:
        x_np = np.random.randn(B, C, H, W).astype(np.float32)
        x = _RT.from_numpy(x_np)

        gms = gpu_bench(lambda: max_pool2d(x, [K, K], [S, S], [0, 0]))
        results.append(Result(
            f"MaxPool {B}x{C}x{H}x{W} k{K}",
            "", gms, 0))

        gms = gpu_bench(lambda: avg_pool2d(x, [K, K], [S, S], [0, 0]))
        results.append(Result(
            f"AvgPool {B}x{C}x{H}x{W} k{K}",
            "", gms, 0))

    return results


# ---------------------------------------------------------------------------
# 11. Normalization
# ---------------------------------------------------------------------------

def bench_norms():
    results = []

    bn_configs = [
        (8,  64, 16, 16),
        (32, 64, 8,  8),
        (16, 128, 8, 8),
    ]
    for B, C, H, W in bn_configs:
        x_np = np.random.randn(B, C, H, W).astype(np.float32)
        x = _RT.from_numpy(x_np)
        w = _RT.ones([C], DType.float32)
        b = _RT.zeros([C], DType.float32)
        rm = _RT.zeros([C], DType.float32)
        rv = _RT.ones([C], DType.float32)
        gms = gpu_bench(lambda: batchnorm_forward(x, w, b, rm, rv, 1e-5, 0.1, True))

        def _np_bn(xn=x_np):
            mu = xn.mean(axis=(0, 2, 3), keepdims=True)
            var = xn.var(axis=(0, 2, 3), keepdims=True)
            return (xn - mu) / np.sqrt(var + 1e-5)

        cms = cpu_bench(_np_bn)
        results.append(Result(f"BatchNorm [{B},{C},{H},{W}]", "", gms, cms))

    for dim in [256, 512, 2048]:
        batch = 128
        x_np = np.random.randn(batch, dim).astype(np.float32)
        x = _RT.from_numpy(x_np)
        wl = _RT.ones([dim], DType.float32)
        bl = _RT.zeros([dim], DType.float32)
        gms = gpu_bench(lambda: layernorm_forward(x, wl, bl, dim, 1e-5))

        def _np_ln(xn=x_np, d=dim):
            mu = xn.mean(axis=-1, keepdims=True)
            var = xn.var(axis=-1, keepdims=True)
            return (xn - mu) / np.sqrt(var + 1e-5)

        cms = cpu_bench(_np_ln)
        results.append(Result(f"LayerNorm [{batch},{dim}]", "", gms, cms))

    return results


# ---------------------------------------------------------------------------
# 12. Optimizer Steps
# ---------------------------------------------------------------------------

def bench_optimizers():
    results = []

    for n in [100_000, 1_000_000, 10_000_000]:
        tag = f"{n // 1000}K" if n < 1_000_000 else f"{n // 1_000_000}M"
        p = _RT.from_numpy(np.random.randn(n).astype(np.float32))
        g = _RT.from_numpy(np.random.randn(n).astype(np.float32))
        v = _RT.zeros([n], DType.float32)

        gms = gpu_bench(lambda: sgd_update(p, g, v, lr=0.01, momentum=0.9, weight_decay=1e-4))
        results.append(Result(f"SGD(mom) {tag}", "", gms, 0))

    for n in [100_000, 1_000_000, 10_000_000]:
        tag = f"{n // 1000}K" if n < 1_000_000 else f"{n // 1_000_000}M"
        p = _RT.from_numpy(np.random.randn(n).astype(np.float32))
        g = _RT.from_numpy(np.random.randn(n).astype(np.float32))
        m_t = _RT.zeros([n], DType.float32)
        v_t = _RT.zeros([n], DType.float32)

        gms = gpu_bench(lambda: adam_update(p, g, m_t, v_t, lr=1e-3, step=100))
        results.append(Result(f"Adam {tag}", "", gms, 0))

        gms = gpu_bench(lambda: adamw_update(p, g, m_t, v_t, lr=1e-3, weight_decay=0.01, step=100))
        results.append(Result(f"AdamW {tag}", "", gms, 0))

    return results


# ---------------------------------------------------------------------------
# 13. Tensor Operations
# ---------------------------------------------------------------------------

def bench_tensor_ops():
    results = []

    n = 2048
    x_np = np.random.randn(n, n).astype(np.float32)
    x = _RT.from_numpy(x_np)

    gms = gpu_bench(lambda: x.transpose(0, 1))
    cms = cpu_bench(lambda: x_np.T.copy())
    results.append(Result(f"Transpose {n}x{n}", "", gms, cms))

    xt = x.transpose(0, 1)
    gms = gpu_bench(lambda: xt.contiguous())
    results.append(Result(f"Contiguous {n}x{n}", "", gms, 0))

    gms = gpu_bench(lambda: _RT.zeros([n, n], DType.float32))
    results.append(Result(f"Zeros {n}x{n}", "", gms, 0))

    gms = gpu_bench(lambda: _RT.ones([n, n], DType.float32))
    results.append(Result(f"Ones {n}x{n}", "", gms, 0))

    gms = gpu_bench(lambda: _RT.randn([n, n], DType.float32))
    results.append(Result(f"Randn {n}x{n}", "", gms, 0))

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_section(title, results):
    print(f"\n{'=' * 78}")
    print(f"  {title}")
    print(f"{'=' * 78}")
    print(f"  {'Kernel':<34s}  {'GPU ms':>8s}  {'CPU ms':>8s}  {'Speedup':>8s}  {'Notes'}")
    print(f"  {'-' * 34}  {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 20}")
    for r in results:
        cpu_str = f"{r.cpu_ms:8.3f}" if r.cpu_ms > 0 else "     N/A"
        sp_str = f"{r.speedup:7.1f}x" if r.cpu_ms > 0 else "     N/A"
        print(f"  {r.name:<34s}  {r.gpu_ms:8.3f}  {cpu_str}  {sp_str}  {r.desc}")


def main():
    info = device_info()
    print("=" * 78)
    print("  cuDeep Kernel Benchmark Suite")
    print("=" * 78)
    print(f"  Device          : {info['name']}")
    print(f"  Compute Cap     : {info['compute_capability']}")
    print(f"  SMs             : {info['multiprocessors']}")
    print(f"  VRAM            : {info['total_memory_mb']} MB")
    print(f"  GPU Clock       : {info['clock_rate_mhz']} MHz")
    print(f"  Mem Clock       : {info['memory_clock_rate_mhz']} MHz")
    print(f"  Mem Bus Width   : {info['memory_bus_width']} bit")
    peak_bw = 2 * info['memory_clock_rate_mhz'] * 1e6 * (info['memory_bus_width'] / 8) / 1e9
    print(f"  Peak Mem BW     : {peak_bw:.0f} GB/s (theoretical)")
    print(f"  NumPy           : {np.__version__} (CPU baseline)")
    print(f"  Precision       : float32")

    sections = [
        ("Matrix Multiplication",  bench_matmul),
        ("Elementwise Operations", bench_elementwise),
        ("Reductions",             bench_reductions),
        ("Activations (forward)",  bench_activations),
        ("Activation Backward",    bench_activation_backward),
        ("Unary Math",             bench_unary_math),
        ("Softmax",                bench_softmax),
        ("Loss Functions",         bench_loss),
        ("Convolution",            bench_conv2d),
        ("Pooling",                bench_pooling),
        ("Normalization",          bench_norms),
        ("Optimizer Steps",        bench_optimizers),
        ("Tensor Operations",      bench_tensor_ops),
    ]

    total_benchmarks = 0
    for title, fn in sections:
        try:
            results = fn()
            print_section(title, results)
            total_benchmarks += len(results)
        except Exception as e:
            print(f"\n  [ERROR] {title}: {e}")

    print(f"\n{'=' * 78}")
    print(f"  Benchmark complete. {total_benchmarks} measurements across {len(sections)} categories.")
    print(f"{'=' * 78}")


if __name__ == "__main__":
    main()
