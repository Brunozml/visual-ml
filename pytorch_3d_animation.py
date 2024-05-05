import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_swiss_roll
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

np.random.seed(1337)
torch.manual_seed(1337)

# Create Swiss roll dataset
X, color = make_swiss_roll(n_samples=1000, noise=0.2)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor((color > color.mean()).astype(np.float32)) * 2 - 1  # make y be -1 or 1

# Define the model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x).squeeze(1)

model = MLP()

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

# Optimization
optimizer = torch.optim.SGD(model.parameters(), lr=1.0) # before lr was 1.0

# Define the learning rate scheduler
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

# Create a 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# Create a scatter plot of the data points
scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, s=40, cmap=plt.cm.Spectral)

# Update function for the animation
def update(i):
    ax.clear()

    total_loss, acc = loss(model, X, y)

    optimizer.zero_grad()
    total_loss.backward()

    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = 1.0 - 0.9 * i / 200

    optimizer.step()

    # Decay the learning rate
    scheduler.step()

    print(f"step {i} loss {total_loss.item()}, accuracy {acc.item()*100}%")

    # Update the colors based on the current predictions
    scores = model(X)
    colors = (scores > 0).float().numpy()

    # Add the original categories as smaller, lighter points
    ax.scatter(X[y==1, 0], X[y==1, 1], X[y==1, 2], c='blue', s=10, marker='o', alpha=0.8)
    ax.scatter(X[y==-1, 0], X[y==-1, 1], X[y==-1, 2], c='red', s=10, marker='s', alpha=0.8)

    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors, s=30, cmap=plt.cm.Spectral, alpha= 0.4)

    return scatter,

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=200, repeat=False, blit=True)

# Save the animation with a lower fps
ani.save('pytorch-3d-animation.gif', writer=PillowWriter(fps=10))
