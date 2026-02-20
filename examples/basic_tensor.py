"""Basic tensor operations with cuDeep."""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "python"))
sys.path.insert(0, str(_root / "build"))

import numpy as np


def main():
    from cuDeep import Tensor, DType

    # Create tensors
    a = Tensor.zeros([3, 3])
    b = Tensor.ones([3, 3])
    print("a:", a)
    print("b:", b)

    # Elementwise ops
    c = a + b
    print("a + b:", c)

    # From/to NumPy
    np_arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    t = Tensor.from_numpy(np_arr)
    print("from numpy:", t)
    print("back to numpy:", t.numpy())

    # Matmul
    x = Tensor.from_numpy(np.eye(4, dtype=np.float32))
    y = Tensor.from_numpy(np.arange(16, dtype=np.float32).reshape(4, 4))
    z = x.matmul(y)
    print("I @ arange(16).reshape(4,4):", z.numpy())


if __name__ == "__main__":
    main()
