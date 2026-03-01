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

Head-to-head against NVIDIA's own libraries on **Jetson Orin** (SM 8.7, 8 SMs, CUDA 12.2, cuDNN 9.3).

### SGEMM (FP32) vs cuBLAS — 8/8 wins

| Size | cuDeep | cuBLAS | Speedup |
|------|-------:|-------:|--------:|
| 256x256x256 | 0.030 ms | 0.052 ms | **1.76x faster** |
| 512x512x512 | 0.114 ms | 0.266 ms | **2.33x faster** |
| 1024x1024x1024 | 0.812 ms | 1.830 ms | **2.25x faster** |
| 2048x2048x2048 | 6.780 ms | 14.282 ms | **2.11x faster** |
| 4096x4096x4096 | 53.581 ms | 113.560 ms | **2.12x faster** |
| 128x4096x512 | 0.257 ms | 0.519 ms | **2.02x faster** |
| 4096x128x512 | 0.270 ms | 0.521 ms | **1.93x faster** |
| 1024x1024x4096 | 3.067 ms | 7.136 ms | **2.33x faster** |

> cuDeep's tiled SGEMM with double-buffered software pipelining and `cp.async` delivers **~2x the throughput** of cuBLAS `cublasSgemm` across all tested sizes.

### Activations vs cuDNN — 12/12 wins

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

### Reductions vs cuBLAS — 6/6 wins

| Op | Size | cuDeep | cuBLAS | Speedup |
|----|------|-------:|-------:|--------:|
| Sum | 128K | 0.018 ms | 0.048 ms | **2.67x faster** |
| Max | 128K | 0.019 ms | 0.048 ms | **2.57x faster** |
| Sum | 1M | 0.072 ms | 0.138 ms | **1.91x faster** |
| Max | 1M | 0.072 ms | 0.124 ms | **1.72x faster** |
| Sum | 8M | 0.518 ms | 0.713 ms | **1.38x faster** |
| Max | 8M | 0.515 ms | 0.570 ms | **1.11x faster** |

> Warp-shuffle tree reductions with vectorized global loads — **up to 2.7x faster** than cuBLAS `sasum`/`isamax`.

### Softmax vs cuDNN — 2 wins, 4 losses

| Size | cuDeep | cuDNN | Result |
|------|-------:|------:|--------|
| 32x32768 | 0.287 ms | 0.362 ms | **1.26x faster** |
| 256x1024 | 0.037 ms | 0.041 ms | **1.10x faster** |
| 128x512 | 0.013 ms | 0.009 ms | 1.50x slower |
| 1024x1024 | 0.174 ms | 0.146 ms | 1.19x slower |

> cuDeep wins on large inner dimensions; cuDNN has the edge on mid-range sizes.

### Areas Under Active Optimization

| Category | vs | Status |
|----------|----|--------|
| TF32 Tensor Core GEMM | cuBLAS TF32 | ~50% of cuBLAS speed — pipeline depth & swizzling WIP |
| Conv2d (im2col + GEMM) | cuDNN | 21-77% of cuDNN — limited by underlying GEMM; Winograd planned |
| BatchNorm (fused) | cuDNN | 61-91% of cuDNN — fused Welford single-pass kernel |
| Pooling | cuDNN | ~90% of cuDNN — vectorization planned |

### Overall: 28 wins / 24 losses / 2 ties across 54 benchmarks

cuDeep **dominates on pure compute** (SGEMM, activations, reductions) and is **closing the gap** on algorithm-heavy ops (conv, batchnorm) where cuDNN leverages Winograd and multi-stage autotuning.

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
