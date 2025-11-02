#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Comparison script (v2, with extrapolation)

Goal:
- Show that frequency/Laplace-based neurons are more stable across runs
- AND that they extrapolate better out-of-domain than a plain MLP.

Models:
- A: Laplace with fixed bases  (convex in amplitudes)
- B: Laplace with learnable bases (still structured, but non-convex)
- C: Plain MLP

Train domain: t ∈ [0, 5]
Test  domain: t ∈ [5, 10]
"""

import math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ============================================================
# 1. Target signal
# ============================================================
def make_signal(T=200, t_min=0.0, t_max=5.0, device="cpu"):
    """
    Slightly nontrivial damped multi-frequency signal
    y(t) = e^{-0.2 t} cos(3t + 0.3) + 0.4 e^{-0.5 t} sin(7t + 1.0)
    """
    t = torch.linspace(t_min, t_max, T, device=device)
    y = (
        torch.exp(-0.2 * t) * torch.cos(3.0 * t + 0.3)
        + 0.4 * torch.exp(-0.5 * t) * torch.sin(7.0 * t + 1.0)
    )
    return t, y

# ============================================================
# 2. Models
# ============================================================
class LaplaceFixed(nn.Module):
    """
    y_hat(t) = sum_k a_k * e^{-s_k t} * cos(w_k t) + b_k * e^{-s_k t} * sin(w_k t)
    with fixed frequencies/dampings.
    """
    def __init__(self, K=16, s_min=0.05, s_max=1.0):
        super().__init__()
        self.K = K
        # frequencies on a grid
        w = torch.linspace(1.0, 8.0, K) * (math.pi / 2.0)
        # damping on a grid
        s = torch.linspace(s_min, s_max, K)
        self.register_buffer("w", w)
        self.register_buffer("s", s)
        # learnable amplitudes
        self.a = nn.Parameter(torch.zeros(K))
        self.b = nn.Parameter(torch.zeros(K))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, t):
        t = t.view(-1, 1)                      # [T,1]
        decay = torch.exp(-t * self.s.view(1, -1))     # [T,K]
        cos_term = torch.cos(t * self.w.view(1, -1))   # [T,K]
        sin_term = torch.sin(t * self.w.view(1, -1))   # [T,K]
        y = decay * (cos_term * self.a.view(1, -1) + sin_term * self.b.view(1, -1))
        return y.sum(dim=1) + self.bias


class LaplaceLearnable(nn.Module):
    """
    Learnable frequencies and dampings.
    """
    def __init__(self, K=16):
        super().__init__()
        self.K = K
        # init around plausible values
        self.log_s = nn.Parameter(torch.full((K,), -1.5))     # softplus -> ~0.2
        self.w = nn.Parameter(torch.linspace(1.0, 8.0, K) * (math.pi / 2.0))
        self.a = nn.Parameter(torch.zeros(K))
        self.b = nn.Parameter(torch.zeros(K))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, t):
        t = t.view(-1, 1)
        s = F.softplus(self.log_s).view(1, -1)
        decay = torch.exp(-t * s)
        cos_term = torch.cos(t * self.w.view(1, -1))
        sin_term = torch.sin(t * self.w.view(1, -1))
        y = decay * (cos_term * self.a.view(1, -1) + sin_term * self.b.view(1, -1))
        return y.sum(dim=1) + self.bias


class MLP(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, t):
        t = t.view(-1, 1)
        return self.net(t).view(-1)

# ============================================================
# 3. Train / eval utils
# ============================================================
def train_model(model, t_train, y_train, epochs=800, lr=1e-3, wd=0.0):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        y_pred = model(t_train)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        opt.step()
    with torch.no_grad():
        final_loss = loss_fn(model(t_train), y_train).item()
    return final_loss, model

def eval_model(model, t_test, y_test):
    with torch.no_grad():
        y_pred = model(t_test)
        loss = F.mse_loss(y_pred, y_test).item()
    return loss, y_pred

# ============================================================
# 4. Experiment loop
# ============================================================
def run_experiment(n_runs=20, device="cpu"):
    # train domain
    t_train, y_train = make_signal(T=200, t_min=0.0, t_max=5.0, device=device)
    # test/extrap domain
    t_test, y_test   = make_signal(T=200, t_min=5.0, t_max=10.0, device=device)

    results = {
        "fixed_train": [],
        "fixed_test": [],
        "learn_train": [],
        "learn_test": [],
        "mlp_train": [],
        "mlp_test": [],
    }

    for i in range(n_runs):
        # on varie vraiment l'init d'un run à l'autre
        torch.manual_seed(1234 + 10*i)
        np.random.seed(1234 + 10*i)
        random.seed(1234 + 10*i)

        # ---------------- A. Laplace FIXED ----------------
        lap_fixed = LaplaceFixed(K=16).to(device)
        fixed_train_loss, lap_fixed = train_model(lap_fixed, t_train, y_train,
                                                  epochs=600, lr=3e-3)
        fixed_test_loss, _ = eval_model(lap_fixed, t_test, y_test)
        results["fixed_train"].append(fixed_train_loss)
        results["fixed_test"].append(fixed_test_loss)

        # ---------------- B. Laplace LEARNABLE ------------
        # on redécale la seed pour casser la symétrie
        torch.manual_seed(8888 + 10*i)
        lap_learn = LaplaceLearnable(K=16).to(device)
        learn_train_loss, lap_learn = train_model(lap_learn, t_train, y_train,
                                                  epochs=800, lr=2e-3)
        learn_test_loss, _ = eval_model(lap_learn, t_test, y_test)
        results["learn_train"].append(learn_train_loss)
        results["learn_test"].append(learn_test_loss)

        # ---------------- C. MLP ---------------------------
        torch.manual_seed(9999 + 10*i)
        mlp = MLP(hidden=64).to(device)
        mlp_train_loss, mlp = train_model(mlp, t_train, y_train,
                                          epochs=800, lr=2e-3)
        mlp_test_loss, _ = eval_model(mlp, t_test, y_test)
        results["mlp_train"].append(mlp_train_loss)
        results["mlp_test"].append(mlp_test_loss)

        print(f"Run {i+1:2d}/{n_runs} | "
              f"Fix train={fixed_train_loss:.6f} test={fixed_test_loss:.6f} | "
              f"Learn train={learn_train_loss:.6f} test={learn_test_loss:.6f} | "
              f"MLP train={mlp_train_loss:.6f} test={mlp_test_loss:.6f}")

    return results

# ============================================================
# 5. Pretty printing
# ============================================================
def print_stats(name, arr):
    arr = np.array(arr)
    print(f"{name}:")
    print(f"  mean  = {arr.mean():.6f}")
    print(f"  std   = {arr.std():.6f}")
    print(f"  min   = {arr.min():.6f}")
    print(f"  max   = {arr.max():.6f}")

# ============================================================
# 6. Main
# ============================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    results = run_experiment(n_runs=20, device=device)

    print("\n=== TRAIN domain (0 → 5) ===")
    print_stats("Laplace fixed",  results["fixed_train"])
    print_stats("Laplace learned",results["learn_train"])
    print_stats("MLP",            results["mlp_train"])

    print("\n=== TEST / EXTRAP domain (5 → 10) ===")
    print_stats("Laplace fixed",  results["fixed_test"])
    print_stats("Laplace learned",results["learn_test"])
    print_stats("MLP",            results["mlp_test"])

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------
    plt.figure(figsize=(11,5))
    bins = 8

    # Train histogram
    plt.subplot(1,2,1)
    plt.hist(results["fixed_train"],  bins=bins, alpha=0.7, label="Laplace fixed (train)")
    plt.hist(results["learn_train"],  bins=bins, alpha=0.7, label="Laplace learned (train)")
    plt.hist(results["mlp_train"],    bins=bins, alpha=0.7, label="MLP (train)")
    plt.title("Train loss distribution")
    plt.xlabel("MSE train")
    plt.ylabel("count")
    plt.legend()

    # Test / extrap histogram
    plt.subplot(1,2,2)
    plt.hist(results["fixed_test"],  bins=bins, alpha=0.7, label="Laplace fixed (test)")
    plt.hist(results["learn_test"],  bins=bins, alpha=0.7, label="Laplace learned (test)")
    plt.hist(results["mlp_test"],    bins=bins, alpha=0.7, label="MLP (test)")
    plt.title("Extrapolation loss distribution (5→10)")
    plt.xlabel("MSE test")
    plt.ylabel("count")
    plt.legend()

    plt.tight_layout()
    plt.show()
