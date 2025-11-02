#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spatio-Temporal Laplace Gesture — 1D spatial interpolation
----------------------------------------------------------
Idée :
- on enregistre un geste souris (t_i, x_i, y_i)
- on le reparamètre par longueur d'arc → u_i ∈ [0,1]
- on entraîne un modèle Laplace multi-bandes qui prend (u, t) et prédit (x, y)
- ensuite : on peut mettre PLUS de u → interpolation spatiale
- et on peut mettre PLUS de t → extrapolation temporelle

C'est une version "perceptron spatio-temporel" mais avec
une base spatiale 1D (le long du geste) au lieu d'une base sin/cos en x.
"""

import time, math, argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib
try:
    matplotlib.use("MacOSX")
except Exception:
    pass
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button


# -----------------------------------------------------------
# 0. Record souris
# -----------------------------------------------------------
def record_mouse(duration=8.0):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_title(f"Move mouse here for {duration}s")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    xs, ys, ts = [], [], []
    start = time.time()
    (dot,) = ax.plot([0.5], [0.5], "ro", ms=6)

    def on_move(event):
        if event.inaxes != ax:
            return
        now = time.time() - start
        xs.append(float(event.xdata))
        ys.append(float(event.ydata))
        ts.append(float(now))
        dot.set_data([event.xdata], [event.ydata])
        fig.canvas.draw_idle()

    cid = fig.canvas.mpl_connect("motion_notify_event", on_move)

    while time.time() - start < duration:
        plt.pause(0.01)

    fig.canvas.mpl_disconnect(cid)
    plt.close(fig)

    xs = np.array(xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)
    ts = np.array(ts, dtype=np.float32)
    return ts, xs, ys


# -----------------------------------------------------------
# 1. Reparamètre le geste en u ∈ [0,1]
#    u = longueur d'arc cumulée / longueur totale
# -----------------------------------------------------------
def build_arc_length(xs, ys):
    if len(xs) < 2:
        return np.zeros_like(xs)
    d = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
    s = np.concatenate([[0.0], np.cumsum(d)])
    total = s[-1] if s[-1] > 0 else 1.0
    u = s / total
    return u.astype(np.float32)


# -----------------------------------------------------------
# 2. Modèle Laplace spatio-temporel (u,t) -> (x,y)
#    - 3 banques temporelles comme tu faisais
#    - mais CHAQUE banque dépend aussi de u
#    - on implémente ça avec un petit MLP sur u
# -----------------------------------------------------------
class LaplaceSTGesture(nn.Module):
    def __init__(
        self,
        duration: float,
        # banks
        Kt_low=32, fmin_low=0.2, fmax_low=4.0,
        Kt_mid=32, fmin_mid_hz=0.5, fmax_mid_hz=6.0,
        Kt_high=16, fmin_high_hz=6.0, fmax_high_hz=20.0,
        hidden_u=32,
    ):
        super().__init__()
        self.duration = float(duration)

        # ----- banque lente (par durée)
        self.Kt_low = Kt_low
        if Kt_low > 0:
            freqs_low = torch.linspace(fmin_low, fmax_low, Kt_low)
            omega_low = 2 * math.pi * freqs_low / self.duration
            self.register_buffer("omega_low", omega_low)
            self.raw_s_low = nn.Parameter(torch.full((Kt_low,), -1.2))
            self.A_low = nn.Linear(hidden_u, 2 * Kt_low)  # produira (real, imag) pour x,y
        else:
            self.register_buffer("omega_low", torch.empty(0))
            self.raw_s_low = nn.Parameter(torch.empty(0))
            self.A_low = nn.Identity()

        # ----- banque moyenne (Hz réels)
        self.Kt_mid = Kt_mid
        if Kt_mid > 0:
            freqs_mid = torch.linspace(fmin_mid_hz, fmax_mid_hz, Kt_mid)
            omega_mid = 2 * math.pi * freqs_mid
            self.register_buffer("omega_mid", omega_mid)
            self.raw_s_mid = nn.Parameter(torch.full((Kt_mid,), -1.6))
            self.A_mid = nn.Linear(hidden_u, 2 * Kt_mid)
        else:
            self.register_buffer("omega_mid", torch.empty(0))
            self.raw_s_mid = nn.Parameter(torch.empty(0))
            self.A_mid = nn.Identity()

        # ----- banque rapide (Hz réels)
        self.Kt_high = Kt_high
        if Kt_high > 0:
            freqs_high = torch.linspace(fmin_high_hz, fmax_high_hz, Kt_high)
            omega_high = 2 * math.pi * freqs_high
            self.register_buffer("omega_high", omega_high)
            self.raw_s_high = nn.Parameter(torch.full((Kt_high,), -2.0))
            self.A_high = nn.Linear(hidden_u, 2 * Kt_high)
        else:
            self.register_buffer("omega_high", torch.empty(0))
            self.raw_s_high = nn.Parameter(torch.empty(0))
            self.A_high = nn.Identity()

        # encodeur spatial (sur u)
        self.u_enc = nn.Sequential(
            nn.Linear(1, hidden_u),
            nn.ReLU(),
            nn.Linear(hidden_u, hidden_u),
            nn.ReLU(),
        )

        # offset global
        self.base_xy = nn.Parameter(torch.zeros(2))

    def _temporal_bank_duration(self, t, omega, raw_s):
        if omega.numel() == 0:
            return torch.zeros(t.shape[0], 0, device=t.device)
        T = t.shape[0]
        tt = t.view(T, 1)
        s = 0.05 + F.softplus(raw_s)
        phase = tt * omega.view(1, -1)
        decay = torch.exp(-tt * s.view(1, -1))
        exp_iwt = torch.complex(torch.cos(phase), torch.sin(phase))
        return decay * exp_iwt  # [T, K]

    def _temporal_bank_hz(self, t, omega, raw_s):
        if omega.numel() == 0:
            return torch.zeros(t.shape[0], 0, device=t.device)
        T = t.shape[0]
        tt = t.view(T, 1)
        s = 0.02 + F.softplus(raw_s)
        phase = tt * omega.view(1, -1)
        decay = torch.exp(-tt * s.view(1, -1))
        exp_iwt = torch.complex(torch.cos(phase), torch.sin(phase))
        return decay * exp_iwt

    def forward(self, t: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        t: [N] temps
        u: [N] position le long du geste (0..1)
        """
        t = t.float()
        u = u.view(-1, 1).float()

        # encode u (même spatial pour les 3 banques)
        u_feat = self.u_enc(u)  # [N, hidden_u]

        # temporel
        low_t = self._temporal_bank_duration(t, self.omega_low, self.raw_s_low)  # [N, K_low]
        mid_t = self._temporal_bank_hz(t, self.omega_mid, self.raw_s_mid)        # [N, K_mid]
        high_t = self._temporal_bank_hz(t, self.omega_high, self.raw_s_high)     # [N, K_high]

        # amplitudes dépendantes de u
        out = 0.0
        N = t.shape[0]

        if self.Kt_low > 0:
            a_low = self.A_low(u_feat)  # [N, 2*K_low]
            a_low = a_low.view(N, 2, self.Kt_low)  # [N,2,K]
            A_low = torch.complex(a_low[:, 0, :], a_low[:, 1, :])  # [N,K]
            xy_low = torch.einsum("nk,nk->n", low_t, A_low)  # c'est un scalaire complexe
            # on va faire mieux : une sortie 2D
            # astuce : on recalcule 2 vecteurs indépendants
            # (pour rester simple on duplique)
            xy_low = torch.stack([xy_low.real, xy_low.imag], dim=-1)  # [N,2]
            out = out + xy_low

        if self.Kt_mid > 0:
            a_mid = self.A_mid(u_feat).view(N, 2, self.Kt_mid)
            A_mid = torch.complex(a_mid[:, 0, :], a_mid[:, 1, :])
            xy_mid = torch.einsum("nk,nk->n", mid_t, A_mid)
            xy_mid = torch.stack([xy_mid.real, xy_mid.imag], dim=-1)
            out = out + xy_mid

        if self.Kt_high > 0:
            a_high = self.A_high(u_feat).view(N, 2, self.Kt_high)
            A_high = torch.complex(a_high[:, 0, :], a_high[:, 1, :])
            xy_high = torch.einsum("nk,nk->n", high_t, A_high)
            xy_high = torch.stack([xy_high.real, xy_high.imag], dim=-1)
            out = out + xy_high

        out = out + self.base_xy.view(1, 2)
        return out  # [N,2]


# -----------------------------------------------------------
# 3. Train
# -----------------------------------------------------------
def train_model(
    model,
    t_torch,
    u_torch,
    xy_torch,
    n_epochs=2000,
    lr=2e-3,
    device="cpu",
):
    model.to(device)
    t_torch = t_torch.to(device)
    u_torch = u_torch.to(device)
    xy_torch = xy_torch.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(1, n_epochs + 1):
        opt.zero_grad()
        xy_pred = model(t_torch, u_torch)
        loss = F.mse_loss(xy_pred, xy_torch)
        loss.backward()
        opt.step()

        if ep % max(1, n_epochs // 20) == 0 or ep == 1:
            print(f"[train] epoch {ep:4d} | loss={loss.item():.6f}")

    return model


# -----------------------------------------------------------
# 4. Visualisation : 3 graphes
#    - 1 : geste original
#    - 2 : repro stricte (mêmes t, mêmes u) → doit coller
#    - 3 : u DENSIFIÉ (interpolation spatiale)
# -----------------------------------------------------------
def replay(ts, xs, ys, model, fps=60):
    # data d'origine
    xy_rec = np.stack([xs, ys], axis=-1)
    N = xy_rec.shape[0]
    t0, t1 = float(ts[0]), float(ts[-1])

    # u original
    u = build_arc_length(xs, ys)
    t_torch = torch.from_numpy(ts).float()
    u_torch = torch.from_numpy(u).float()

    model.eval()
    with torch.no_grad():
        xy_pred = model(t_torch, u_torch).cpu().numpy()

    # u densifié (interpolation spatiale)
    N_interp = N * 3
    u_dense = np.linspace(0.0, 1.0, N_interp, dtype=np.float32)
    t_dense = np.linspace(t0, t1, N_interp, dtype=np.float32)
    with torch.no_grad():
        xy_dense = model(torch.from_numpy(t_dense), torch.from_numpy(u_dense)).cpu().numpy()

    # figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Original vs ST-Laplace vs Spatial Interp")

    # limites
    x_min = min(xs.min(), xy_pred[:, 0].min(), xy_dense[:, 0].min())
    x_max = max(xs.max(), xy_pred[:, 0].max(), xy_dense[:, 0].max())
    y_min = min(ys.min(), xy_pred[:, 1].min(), xy_dense[:, 1].min())
    y_max = max(ys.max(), xy_pred[:, 1].max(), xy_dense[:, 1].max())

    for ax in (ax1, ax2, ax3):
        ax.set_xlim(x_min - 0.02, x_max + 0.02)
        ax.set_ylim(y_min - 0.02, y_max + 0.02)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

    ax1.set_title("Recorded")
    ax2.set_title("ST-Laplace (same t,u)")
    ax3.set_title("ST-Laplace (u densifié)")

    (p1,) = ax1.plot([], [], "o", color="tab:blue")
    (p2,) = ax2.plot([], [], "o", color="tab:red")
    (p3,) = ax3.plot([], [], "o", color="tab:green")

    trail1, = ax1.plot([], [], "-", color="tab:blue", alpha=0.6)
    trail2, = ax2.plot([], [], "-", color="tab:red", alpha=0.6)
    trail3, = ax3.plot([], [], "-", color="tab:green", alpha=0.6)

    ANIM = [None]
    max_frames = max(N, N, N_interp)

    def init():
        return p1, p2, p3, trail1, trail2, trail3

    def update(frame):
        # 1
        if frame < N:
            p1.set_data([xy_rec[frame, 0]], [xy_rec[frame, 1]])
            trail1.set_data(xy_rec[: frame + 1, 0], xy_rec[: frame + 1, 1])
        else:
            p1.set_data([xy_rec[-1, 0]], [xy_rec[-1, 1]])
            trail1.set_data(xy_rec[:, 0], xy_rec[:, 1])

        # 2
        if frame < N:
            p2.set_data([xy_pred[frame, 0]], [xy_pred[frame, 1]])
            trail2.set_data(xy_pred[: frame + 1, 0], xy_pred[: frame + 1, 1])
        else:
            p2.set_data([xy_pred[-1, 0]], [xy_pred[-1, 1]])
            trail2.set_data(xy_pred[:, 0], xy_pred[:, 1])

        # 3
        if frame < N_interp:
            p3.set_data([xy_dense[frame, 0]], [xy_dense[frame, 1]])
            trail3.set_data(xy_dense[: frame + 1, 0], xy_dense[: frame + 1, 1])
        else:
            p3.set_data([xy_dense[-1, 0]], [xy_dense[-1, 1]])
            trail3.set_data(xy_dense[:, 0], xy_dense[:, 1])

        return p1, p2, p3, trail1, trail2, trail3

    interval = 1000 // fps

    def start_anim(event=None):
        if ANIM[0] is not None:
            try:
                ANIM[0].event_source.stop()
            except Exception:
                pass
        anim = FuncAnimation(
            fig,
            update,
            frames=max_frames,
            init_func=init,
            interval=interval,
            blit=False,
            repeat=False,
        )
        ANIM[0] = anim
        fig.canvas.draw_idle()

    ax_button = plt.axes([0.88, 0.02, 0.10, 0.05])
    btn = Button(ax_button, "Restart")
    btn.on_clicked(start_anim)

    start_anim()
    plt.show()


# -----------------------------------------------------------
# 5. Main
# -----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-sec", type=float, default=8.0)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=2e-3)
    # tailles de banks (tu peux jouer)
    parser.add_argument("--Kt-low", type=int, default=32)
    parser.add_argument("--Kt-mid", type=int, default=32)
    parser.add_argument("--Kt-high", type=int, default=16)
    parser.add_argument("--fmax-high-hz", type=float, default=20.0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    print(f">>> STEP 1: record {args.record_sec}s …")
    ts, xs, ys = record_mouse(duration=args.record_sec)
    print(f"Recorded {len(ts)} samples.")

    # construit u
    u = build_arc_length(xs, ys)

    t_torch = torch.from_numpy(ts).float()
    u_torch = torch.from_numpy(u).float()
    xy_torch = torch.from_numpy(np.stack([xs, ys], axis=-1)).float()

    duration = float(ts.max())

    print(">>> STEP 2: build model …")
    model = LaplaceSTGesture(
        duration=duration,
        Kt_low=args.Kt_low,
        Kt_mid=args.Kt_mid,
        Kt_high=args.Kt_high,
        fmax_high_hz=args.fmax_high_hz,
    )

    print(">>> STEP 3: train …")
    model = train_model(
        model,
        t_torch,
        u_torch,
        xy_torch,
        n_epochs=args.epochs,
        lr=args.lr,
        device=device,
    )

    print(">>> STEP 4: replay …")
    replay(ts, xs, ys, model, fps=60)


if __name__ == "__main__":
    main()
