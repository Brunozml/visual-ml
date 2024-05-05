import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

from sklearn.datasets import make_moons

np.random.seed(1337)
random.seed(1337)
torch.manual_seed(1337)

X, y = make_moons(n_samples=100, noise=0.1)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32) * 2 - 1  # make y be -1 or 1

# Define the model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x).squeeze(1)

model = MLP()
print(model)
print("number of parameters", sum(p.numel() for p in model.parameters()))

# Loss function
def loss(model, X, y, batch_size=None, alpha=1e-4):
    if batch_size is None:
        Xb, yb = X, y
    else:
        ri = torch.randperm(X.size(0))[:batch_size]
        Xb, yb = X[ri], y[ri]

    scores = model(Xb)

    losses = F.relu(1 - yb * scores)
    data_loss = losses.mean()

    reg_loss = alpha * sum(p.pow(2.0).sum() for p in model.parameters())

    total_loss = data_loss + reg_loss

    accuracy = ((yb > 0) == (scores > 0)).float().mean()

    return total_loss, accuracy

total_loss, acc = loss(model, X, y)
print(total_loss.item(), acc.item())

# Optimization
fig, ax = plt.subplots()

optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

def update(i):
    ax.clear()

    total_loss, acc = loss(model, X, y)

    optimizer.zero_grad()
    total_loss.backward()

    for param_group in optimizer.param_groups:
        param_group['lr'] = 1.0 - 0.9 * i / 100

    optimizer.step()

    print(f"step {i} loss {total_loss.item()}, accuracy {acc.item()*100}%")

    h = 0.25
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Xmesh = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    scores = model(Xmesh)
    Z = (scores > 0).numpy()
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    ax.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())

ani = animation.FuncAnimation(fig, update, frames=100, repeat=False)

# Save the animation
ani.save('pytorch-animation-v1.gif', writer=PillowWriter(fps=60))
