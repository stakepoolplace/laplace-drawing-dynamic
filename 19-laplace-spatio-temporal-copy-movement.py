#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Laplace spatio-temporal neuron demo
-----------------------------------

Goal:
- Learn a complex 2D motion described as a chain of relative vectors
  (root fixed, each joint is relative to the previous one)
- Show that a Laplace spatio-temporal unit can reproduce it in real time

Deps: torch, numpy, matplotlib
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
try:
    matplotlib.use("MacOSX")
except Exception:
    pass
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ============================================================
# 1. Synthetic complex motion (relative vectors)
# ============================================================
def make_complex_motion(T=400, duration=4.0, n_segments=4):
    """
    Returns:
        rel_vecs: [T, n_segments, 2]  relative vectors (dx, dy)
        t: [T] time array
    Skeleton:
        p0 = (0,0) fixed
        p1 = p0 + v1(t)
        p2 = p1 + v2(t)
        ...
    We make each segment oscillate with different freq/phase,
    so motion is nontrivial and non purely periodic.
    """
    t = np.linspace(0, duration, T)
    rel = np.zeros((T, n_segments, 2), dtype=np.float32)

    # segment 1 : grand balayage
    rel[:, 0, 0] = 0.8 + 0.2*np.sin(2*np.pi*t/duration * 1.0)
    rel[:, 0, 1] = 0.2*np.cos(2*np.pi*t/duration * 0.5)

    # segment 2 : petit cercle
    rel[:, 1, 0] = 0.4*np.cos(2*np.pi*t/duration * 1.7 + 0.7)
    rel[:, 1, 1] = 0.4*np.sin(2*np.pi*t/duration * 1.7 + 0.7)

    # segment 3 : vibration amortie
    rel[:, 2, 0] = 0.25*np.exp(-0.4*t) * np.cos(2*np.pi*t/duration * 3.5 + 1.4)
    rel[:, 2, 1] = 0.25*np.exp(-0.4*t) * np.sin(2*np.pi*t/duration * 2.3 + 0.3)

    # segment 4 : mouvement plus lent, style suivi
    rel[:, 3, 0] = 0.3*np.sin(2*np.pi*t/duration * 0.6 + 1.2)
    rel[:, 3, 1] = 0.15*np.sin(2*np.pi*t/duration * 0.9 + 2.0)

    return torch.from_numpy(rel), torch.from_numpy(t).float()


# ============================================================
# 2. Laplace spatio-temporal neuron
#    - shared temporal basis
#    - per-segment complex amplitudes
#    - output: relative vectors
# ============================================================
class LaplaceMotion(nn.Module):
    def __init__(self, n_segments=4, Kt=32, duration=4.0, max_s=1.5):
        super().__init__()
        self.n_segments = n_segments
        self.Kt = Kt
        self.duration = duration
        self.max_s = max_s

        # base des fréquences
        w = 2*math.pi*torch.linspace(0.3, 3.5, Kt) / duration
        self.register_buffer("omega", w)  # [K]

        # damping (à apprendre)
        self.raw_s = nn.Parameter(torch.full((Kt,), -1.5))  # softplus -> ~0.2

        # amplitudes complexes par segment et par axe (x,y)
        # shape: [n_segments, 2, K]
        self.A_real = nn.Parameter(torch.randn(n_segments, 2, Kt) * 0.01)
        self.A_imag = nn.Parameter(torch.randn(n_segments, 2, Kt) * 0.01)

        # offset optionnel sur les vecteurs (pour avoir une pose moyenne)
        self.base_vec = nn.Parameter(torch.zeros(n_segments, 2))

    def forward(self, t_grid):
        """
        t_grid: [T]
        return: rel_vecs [T, n_segments, 2]
        """
        T = t_grid.shape[0]
        t = t_grid.view(T, 1)  # [T,1]

        s = F.softplus(self.raw_s).clamp(max=self.max_s)      # [K]
        w = self.omega                                         # [K]

        # exp(- (s + i w) t)
        decay = torch.exp(-t * s.view(1, -1))                  # [T,K]
        phase = t * w.view(1, -1)                              # [T,K]
        exp_iwt = torch.complex(torch.cos(phase), torch.sin(phase))  # [T,K]
        # full temporal kernel
        kernel = decay * exp_iwt                               # [T,K] complex

        # (A_real + i A_imag) : [S,2,K]
        A = torch.complex(self.A_real, self.A_imag)            # [S,2,K]

        # on veut: [T, S, 2]
        # kernel: [T,K], A: [S,2,K] -> einsum
        rel_complex = torch.einsum("tk,sdk->tsd", kernel, A)   # [T,S,2] complex
        rel = rel_complex.real + self.base_vec.view(1, self.n_segments, 2)
        return rel  # [T,S,2]


# ============================================================
# 3. Utility: rebuild absolute positions from relative vectors
# ============================================================
def rel_to_abs(rel_vecs, root=(0.0, 0.0)):
    """
    rel_vecs: [T, S, 2]
    return: abs_pos: [T, S+1, 2]  (root + each joint)
    """
    T, S, _ = rel_vecs.shape
    root_xy = torch.tensor(root, dtype=rel_vecs.dtype, device=rel_vecs.device).view(1, 1, 2)
    rel_cum = torch.cumsum(rel_vecs, dim=1)  # [T,S,2]
    abs_pos = torch.cat([root_xy.repeat(T, 1, 1), root_xy + rel_cum], dim=1)
    return abs_pos


# ============================================================
# 4. Training
# ============================================================
def train_model(model, t, rel_target, n_epochs=1200, lr=2e-3, print_every=100, device="cpu"):
    model.to(device)
    t = t.to(device)
    rel_target = rel_target.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(1, n_epochs+1):
        opt.zero_grad()
        rel_pred = model(t)              # [T,S,2]
        # loss principale : MSE sur les vecteurs relatifs
        loss_rel = F.mse_loss(rel_pred, rel_target)
        # perte de lissage dans le temps
        diff = rel_pred[1:] - rel_pred[:-1]
        loss_smooth = (diff**2).mean()
        loss = loss_rel + 0.01*loss_smooth
        loss.backward()
        opt.step()
        if ep % print_every == 0 or ep == 1:
            print(f"Epoch {ep:4d} | loss={loss.item():.6f} | rel={loss_rel.item():.6f}")
    return model


# ============================================================
# 5. Visualization / real-time playback
# ============================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # 1) data
    rel_target, t = make_complex_motion(T=400, duration=4.0, n_segments=4)
    # normalize un peu si tu veux
    # 2) model
    model = LaplaceMotion(n_segments=4, Kt=32, duration=4.0)

    # 3) train
    model = train_model(model, t, rel_target, n_epochs=1200, lr=2e-3, device=device)

    # 4) compute prediction once
    model.eval()
    with torch.no_grad():
        rel_pred = model(t.to(device)).cpu()
    abs_target = rel_to_abs(rel_target)
    abs_pred   = rel_to_abs(rel_pred)

    # 5) animation
    T, S1, _ = abs_target.shape   # S1 = n_segments+1
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    fig.suptitle("Laplace spatio-temporal neuron – complex motion reproduction")

    # limits (same for both)
    all_pts = torch.cat([abs_target.reshape(T*S1, 2), abs_pred.reshape(T*S1, 2)], dim=0).numpy()
    xmin, ymin = all_pts.min(axis=0) - 0.2
    xmax, ymax = all_pts.max(axis=0) + 0.2

    for ax in (ax1, ax2):
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)

    ax1.set_title("Target motion")
    ax2.set_title("Laplace neuron output")

    # init lines
    target_lines = []
    pred_lines = []
    for _ in range(S1-1):
        l1, = ax1.plot([], [], 'o-', lw=3, color='tab:blue')
        l2, = ax2.plot([], [], 'o-', lw=3, color='tab:red')
        target_lines.append(l1)
        pred_lines.append(l2)

    # trail (end effector)
    trail1, = ax1.plot([], [], '-', lw=1.5, color='tab:blue', alpha=0.4)
    trail2, = ax2.plot([], [], '-', lw=1.5, color='tab:red', alpha=0.4)
    trail_buf_1 = []
    trail_buf_2 = []

    def init():
        return target_lines + pred_lines + [trail1, trail2]

    def update(frame):
        # target
        pts_t = abs_target[frame].numpy()  # [S1,2]
        pts_p = abs_pred[frame].numpy()    # [S1,2]
        for j in range(S1-1):
            target_lines[j].set_data([pts_t[j,0], pts_t[j+1,0]],
                                    [pts_t[j,1], pts_t[j+1,1]])
            pred_lines[j].set_data([pts_p[j,0], pts_p[j+1,0]],
                                  [pts_p[j,1], pts_p[j+1,1]])

        # trail
        ee_t = pts_t[-1]
        ee_p = pts_p[-1]
        trail_buf_1.append(ee_t)
        trail_buf_2.append(ee_p)
        if len(trail_buf_1) > 200:
            trail_buf_1.pop(0); trail_buf_2.pop(0)
        tb1 = np.array(trail_buf_1)
        tb2 = np.array(trail_buf_2)
        trail1.set_data(tb1[:,0], tb1[:,1])
        trail2.set_data(tb2[:,0], tb2[:,1])

        return target_lines + pred_lines + [trail1, trail2]

    anim = FuncAnimation(fig, update, frames=T, init_func=init,
                         interval=20, blit=False, repeat=True)
    plt.show()


if __name__ == "__main__":
    main()
