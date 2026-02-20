"""End-to-end MLP training with cuDeep autograd.

Demonstrates: model definition, forward pass, loss computation,
automatic backward pass, optimizer step, LR scheduling, and inference.
"""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "python"))
sys.path.insert(0, str(_root / "build"))

import numpy as np
import cuDeep
from cuDeep import Tensor, nn, optim, mse_loss, no_grad
from cuDeep import init, lr_scheduler

np.random.seed(42)

# ---- XOR-like regression (nonlinear) ----
x_np = np.array([
    [0, 0], [0, 1], [1, 0], [1, 1],
    [0.1, 0.1], [0.1, 0.9], [0.9, 0.1], [0.9, 0.9],
], dtype=np.float32)
y_np = np.array([
    [0], [1], [1], [0],
    [0], [1], [1], [0],
], dtype=np.float32)

x = Tensor.from_numpy(x_np)
y = Tensor.from_numpy(y_np)

# ---- Define model ----
model = nn.Sequential(
    nn.Linear(2, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.GELU(),
    nn.Linear(32, 1),
)

for p in model.parameters():
    if len(p.shape()) >= 2:
        init.xavier_uniform_(p)
    else:
        init.zeros_(p)

optimizer = optim.Adam(model.parameters(), lr=5e-3)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-4)

print(f"cuDeep v{cuDeep.__version__} | device: {cuDeep.device_info()['name']}")
print(f"Model params: {sum(p.numel() for p in model.parameters())}")
print()

for step in range(500):
    optimizer.zero_grad()
    pred = model(x)
    loss = mse_loss(pred, y)
    loss.backward()
    optimizer.step()
    scheduler.step()

    if step % 50 == 0:
        print(f"step {step:3d}  loss = {loss.item():.6f}  lr = {optimizer.lr:.6f}")

# ---- Evaluate ----
model.eval()
with no_grad():
    pred = model(x).numpy().flatten()
    print(f"\nPredictions: {np.round(pred, 3)}")
    print(f"Targets:     {y_np.flatten()}")
    print(f"Final loss:  {mse_loss(model(x), y).item():.6f}")

# ---- State dict ----
sd = model.state_dict()
print(f"\nSaved state_dict with {len(sd)} parameters")
print("Training complete!")
