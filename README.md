# cuDeep

Ultra-high performance deep learning library implemented in low-level CUDA with Python bindings.

## Features

- **Custom CUDA kernels** — matmul (tiled), convolutions, activations (ReLU, GELU, SiLU, ...), reductions, loss functions
- **Tensor class** with GPU memory pooling, async streams, and NumPy interop
- **PyBind11 Python bindings** — `import cuDeep`
- **nn.Module system** — Linear, Conv2d, BatchNorm2d, LayerNorm, Sequential, etc.
- **Optimizers** — SGD, Adam, AdamW
- **Mixed precision** and tensor core support (planned)

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
