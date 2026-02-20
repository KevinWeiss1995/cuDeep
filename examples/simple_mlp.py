"""Simple MLP example using cuDeep.nn."""

import numpy as np


def main():
    from cuDeep import Tensor, mse_loss
    from cuDeep.nn import Linear, ReLU, GELU, Sequential

    model = Sequential(
        Linear(784, 256),
        ReLU(),
        Linear(256, 128),
        GELU(),
        Linear(128, 10),
    )

    print(f"Model parameters: {len(model.parameters())}")

    x = Tensor.randn([32, 784])
    out = model(x)
    print(f"Input shape:  {x.shape()}")
    print(f"Output shape: {out.shape()}")

    target = Tensor.zeros([32, 10])
    loss = mse_loss(out, target)
    print(f"MSE Loss: {loss.numpy()[0]:.4f}")

    from cuDeep.optim import SGD
    opt = SGD(model.parameters(), lr=0.001)
    print("\nTraining (5 steps):")
    for step in range(5):
        out = model(x)
        loss = mse_loss(out, target)
        grads = [Tensor.randn(p.shape(), p.dtype()) for p in model.parameters()]
        opt.step(grads)
        print(f"  step {step + 1}: loss = {loss.numpy()[0]:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
