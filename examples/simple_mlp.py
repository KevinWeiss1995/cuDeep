"""Simple MLP example using cuDeep.nn."""

import numpy as np


def main():
    from cuDeep import Tensor
    from cuDeep.nn import Linear, ReLU, Sequential

    model = Sequential(
        Linear(784, 256),
        # ReLU(),  # TODO: wire up activation kernels
        Linear(256, 10),
    )

    print("Model parameters:", len(model.parameters()))

    # Dummy forward pass (will work once kernels are wired)
    # x = Tensor.randn([32, 784])
    # out = model(x)
    # print("Output shape:", out.shape())


if __name__ == "__main__":
    main()
