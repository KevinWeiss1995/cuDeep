# cuDeep

Ultra-high performance deep learning library implemented in low-level CUDA with Python bindings.

## Features

- **Custom CUDA kernels** — matmul (tiled), convolutions, activations (ReLU, GELU, SiLU, ...), reductions, loss functions
- **Tensor class** with GPU memory pooling, async streams, and NumPy interop
- **PyBind11 Python bindings** — `import cuDeep`
- **nn.Module system** — Linear, Conv2d, BatchNorm2d, LayerNorm, Sequential, etc.
- **Optimizers** — SGD, Adam, AdamW
- **Mixed precision** and TF32 tensor core support

## Benchmarks — cuDeep vs cuBLAS / cuDNN

Honest head-to-head against NVIDIA's own libraries on **Jetson Orin** (SM 8.7, 8 SMs, CUDA 12.2, cuDNN 9.3). All comparisons use **matching precision modes** — no mixing TF32 against FP32.

### Where cuDeep wins

**Activations vs cuDNN — 12/12 wins**

| Kernel | Size | cuDeep | cuDNN | Speedup |
|--------|------|-------:|------:|--------:|
| ReLU | 1M | 0.135 ms | 0.151 ms | **1.11x** |
| Sigmoid | 1M | 0.135 ms | 0.157 ms | **1.16x** |
| Tanh | 1M | 0.135 ms | 0.153 ms | **1.13x** |
| SiLU | 1M | 0.135 ms | 0.154 ms | **1.14x** |
| ReLU | 16M | 2.090 ms | 2.249 ms | **1.08x** |
| Sigmoid | 16M | 2.089 ms | 2.326 ms | **1.11x** |
| Tanh | 16M | 2.090 ms | 2.283 ms | **1.09x** |
| SiLU | 16M | 2.090 ms | 2.286 ms | **1.09x** |

> Vectorized `float4` loads/stores with fused compute — consistently **8-16% faster** than cuDNN activations.

**Reductions vs cuBLAS — 6/6 wins**

| Op | Size | cuDeep | cuBLAS | Speedup |
|----|------|-------:|-------:|--------:|
| Sum | 128K | 0.018 ms | 0.048 ms | **2.67x faster** |
| Max | 128K | 0.019 ms | 0.048 ms | **2.57x faster** |
| Sum | 1M | 0.072 ms | 0.138 ms | **1.91x faster** |
| Max | 1M | 0.072 ms | 0.124 ms | **1.72x faster** |
| Sum | 8M | 0.518 ms | 0.713 ms | **1.38x faster** |
| Max | 8M | 0.515 ms | 0.570 ms | **1.11x faster** |

> Warp-shuffle tree reductions with vectorized global loads — **up to 2.7x faster** than cuBLAS `sasum`/`isamax`.

### Where cuDeep is competitive

**SGEMM FP32 (CUDA Core, no Tensor Cores) vs cuBLAS — within 4-17%**

| Size | cuDeep | cuBLAS | Gap |
|------|-------:|-------:|----:|
| 512 | 0.822 ms | 0.794 ms | 1.04x slower |
| 1024 | 6.152 ms | 5.244 ms | 1.17x slower |
| 2048 | 49.44 ms | 44.79 ms | 1.10x slower |
| 4096 | 401.4 ms | 384.8 ms | 1.04x slower |

> Hand-written tiled SGEMM with double-buffered `cp.async` pipelining, reaching **86-96% of cuBLAS FP32** at large sizes. For small matrices (N < 512), cuBLAS's heuristic dispatch has a significant edge.

**Softmax vs cuDNN — mixed (2 wins, 4 losses)**

> cuDeep wins on large inner dimensions (32K+); cuDNN has the edge on mid-range sizes (128-1024).

### Areas under active optimization

| Category | vs | Status |
|----------|----|--------|
| TF32 Tensor Core GEMM | cuBLAS TF32 | 34-72% of cuBLAS TF32 — pipeline depth, swizzling, warp scheduling WIP |
| Conv2d (im2col + GEMM) | cuDNN | 21-77% of cuDNN — bottlenecked by underlying GEMM; Winograd planned for 3x3 |
| BatchNorm (fused Welford) | cuDNN | 61-91% of cuDNN |
| Pooling | cuDNN | ~90% of cuDNN — vectorization planned |

### Methodology

All benchmarks use matching precision modes on both sides. The SGEMM section forces the FP32 CUDA Core path (no Tensor Cores), compared against `cublasSgemm` (also FP32). The TF32 section compares cuDeep's Tensor Core kernel against `cublasGemmEx` with `CUBLAS_COMPUTE_32F_FAST_TF32`. We don't inflate numbers by comparing different precisions.

## Requirements

- CUDA Toolkit 12.2+
- CMake 3.24+
- Python 3.9+
- pybind11

## Build

```bash
# CMake build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)

# Or pip install (editable)
pip install -e .
```

## Usage

```python
import cuDeep
import numpy as np

# Tensor operations
a = cuDeep.Tensor.zeros([3, 3])
b = cuDeep.Tensor.ones([3, 3])
c = a + b

# NumPy interop
arr = np.random.randn(4, 4).astype(np.float32)
t = cuDeep.Tensor.from_numpy(arr)
print(t.numpy())

# Neural network
from cuDeep.nn import Linear, Sequential
model = Sequential(
    Linear(784, 256),
    Linear(256, 10),
)
```

## Running Tests

```bash
# C++ tests (after cmake build)
cd build && ctest --verbose

# Python tests (after pip install)
pytest tests/
```

## License

MIT
