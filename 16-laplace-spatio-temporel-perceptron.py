#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spatio-Temporal Laplace Perceptron
Author : Eric Marchand
Date   : 2025-11-01

Implements a full Laplace Perceptron that folds both time and space:
    Y(x,t) = Re( Σ_k,m A[k,m] * exp(-s[k] * t) * φ_m(x) )

Features:
  1. Synthetic target field Y_true(x,t)
  2. Spatial Laplacian eigenmodes φ_m(x)
  3. Gradient-based training
  4. Visualization of true vs predicted field
"""

import torch, torch.nn as nn, torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --------------------------------------------------------------
# Device setup
# --------------------------------------------------------------
torch.set_default_dtype(torch.float32)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🧠 Using device: {device}")

# --------------------------------------------------------------
# Synthetic data
# --------------------------------------------------------------
# Spatial and temporal grids
X = torch.linspace(0, 2*np.pi, 128)
T = torch.linspace(0, 4*np.pi, 200)

def true_field(x, t):
    """Ground truth spatio-temporal function"""
    return (
        torch.sin(2*x) * torch.cos(1.5*t) * torch.exp(-0.1*t)
      + 0.5 * torch.sin(3*x + 0.4) * torch.cos(3*t + 0.7) * torch.exp(-0.05*t)
    )

Y_true = torch.stack([true_field(X, t) for t in T])  # [T, X]
Y_true = Y_true.to(device)

# --------------------------------------------------------------
# Spatial Laplacian eigenmodes φ_m(x)
# --------------------------------------------------------------
def spatial_modes(x, n_modes):
    """
    Generates approximate Laplacian eigenmodes on [0, 2π]
    φ_m(x) = sin(mx), cos(mx)
    """
    phi = []
    for m in range(1, n_modes//2 + 1):
        phi.append(torch.sin(m*x))
        phi.append(torch.cos(m*x))
    phi = torch.stack(phi, dim=1)  # [X, n_modes]
    phi = phi / torch.norm(phi, dim=0, keepdim=True)
    return phi

N_XMODES = 16
Phi_X = spatial_modes(X, N_XMODES).to(device)  # [X, n_x]

# --------------------------------------------------------------
# Laplace Spatio-Temporal Model
# --------------------------------------------------------------
class LaplaceSpatioTemporal(nn.Module):
    def __init__(self, n_t=32, n_x=N_XMODES):
        super().__init__()
        # complex parameters
        self.s = nn.Parameter(torch.randn(n_t, dtype=torch.cfloat) * 0.05)
        self.A = nn.Parameter(torch.randn(n_t, n_x, dtype=torch.cfloat) * 0.05)

    def forward(self, t, phi_x):
        """
        t: [T] temporal grid
        phi_x: [X, n_x] spatial eigenmodes
        """
        # Temporal exponentials (Laplace domain)
        e_t = torch.exp(-t[:, None] * self.s[None, :])  # [T, n_t]

        # Convert spatial basis to complex
        phi_x = phi_x.to(torch.cfloat)  # [X, n_x]

        # Combine temporal and spatial spectra
        # (bilinear spectral synthesis)
        Y_hat = torch.einsum('tk,km,mx->tx', e_t, self.A, phi_x.T)
        return Y_hat.real

# --------------------------------------------------------------
# Training setup
# --------------------------------------------------------------
model = LaplaceSpatioTemporal(n_t=32).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

T_grid = T.to(device)
Phi_grid = Phi_X.to(device)

EPOCHS = 2000
print("🚀 Training started...\n")

for epoch in range(EPOCHS):
    optimizer.zero_grad()
    Y_pred = model(T_grid, Phi_grid)
    loss = loss_fn(Y_pred, Y_true)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Loss = {loss.item():.8f}")

print("\n✅ Training complete.")
print(f"Final loss: {loss.item():.8f}")

# --------------------------------------------------------------
# Visualization
# --------------------------------------------------------------
Y_pred = model(T_grid, Phi_grid).detach().cpu().numpy()
Y_true_np = Y_true.cpu().numpy()
Tg, Xg = np.meshgrid(T.cpu(), X.cpu(), indexing='ij')

fig = plt.figure(figsize=(12,5))

ax1 = fig.add_subplot(1,2,1, projection='3d')
ax1.plot_surface(Xg, Tg, Y_true_np, cmap='viridis')
ax1.set_title("Ground Truth Y(x,t)")
ax1.set_xlabel("x"); ax1.set_ylabel("t")

ax2 = fig.add_subplot(1,2,2, projection='3d')
ax2.plot_surface(Xg, Tg, Y_pred, cmap='plasma')
ax2.set_title("Laplace Perceptron Prediction")
ax2.set_xlabel("x"); ax2.set_ylabel("t")

plt.tight_layout()
plt.show()
