#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interactive Laplace Spatio-Temporal Demo — 3 banks
--------------------------------------------------
Objectif : mieux séparer les échelles de mouvement
- bank low  : forme globale (fréquences réparties sur la durée)
- bank mid  : mouvements intermédiaires (Hz réels bas)
- bank high : micro-mouvements (Hz réels plus hauts)

⚠️ Comportement demandé :
- si on NE passe PAS --extrap-pct → PAS d'extrapolation
  → le 3e graphe montre juste la prédiction sur les temps bruts (comme le 2e) mais sans clamp
- si on passe --extrap-pct X → extrapolation de X × durée en plus
"""

import time
import math
import argparse
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
from matplotlib.widgets import Button


# ============================================================
# 1. Modèle Laplace 3 banques
# ============================================================
class LaplaceMotion3Banks(nn.Module):
    def __init__(
        self,
        duration: float,
        # banque lente (durée)
        Kt_low: int = 32,
        fmin_low: float = 0.2,
        fmax_low: float = 4.0,
        s_floor_low: float = 0.05,
        max_s_low: float = 2.0,
        # banque moyenne (Hz réels)
        Kt_mid: int = 48,
        fmin_mid_hz: float = 0.5,
        fmax_mid_hz: float = 6.0,
        s_floor_mid: float = 0.02,
        max_s_mid: float = 1.5,
        # banque rapide (Hz réels)
        Kt_high: int = 96,
        fmin_high_hz: float = 6.0,
        fmax_high_hz: float = 40.0,
        s_floor_high: float = 0.0,
        max_s_high: float = 1.0,
    ):
        super().__init__()
        self.duration = float(duration)

        # ----- banque lente (par durée)
        self.Kt_low = Kt_low
        if Kt_low > 0:
            freqs_low = torch.linspace(fmin_low, fmax_low, Kt_low, dtype=torch.float32)
            # même logique que ton code de base : 2π f / duration
            omega_low = 2 * math.pi * freqs_low / self.duration
            self.register_buffer("omega_low", omega_low)
            self.raw_s_low = nn.Parameter(torch.full((Kt_low,), -1.2, dtype=torch.float32))
            self.A_low_real = nn.Parameter(torch.randn(2, Kt_low) * 0.01)
            self.A_low_imag = nn.Parameter(torch.randn(2, Kt_low) * 0.01)
            self.s_floor_low = float(s_floor_low)
            self.max_s_low = float(max_s_low)
        else:
            self.register_buffer("omega_low", torch.empty(0))
            self.raw_s_low = nn.Parameter(torch.empty(0))
            self.A_low_real = nn.Parameter(torch.empty(2, 0))
            self.A_low_imag = nn.Parameter(torch.empty(2, 0))
            self.s_floor_low = 0.0
            self.max_s_low = 0.0

        # ----- banque moyenne (Hz réels)
        self.Kt_mid = Kt_mid
        if Kt_mid > 0:
            freqs_mid = torch.linspace(fmin_mid_hz, fmax_mid_hz, Kt_mid, dtype=torch.float32)
            omega_mid = 2 * math.pi * freqs_mid  # rad/s
            self.register_buffer("omega_mid", omega_mid)
            self.raw_s_mid = nn.Parameter(torch.full((Kt_mid,), -1.6, dtype=torch.float32))
            self.A_mid_real = nn.Parameter(torch.randn(2, Kt_mid) * 0.01)
            self.A_mid_imag = nn.Parameter(torch.randn(2, Kt_mid) * 0.01)
            self.s_floor_mid = float(s_floor_mid)
            self.max_s_mid = float(max_s_mid)
        else:
            self.register_buffer("omega_mid", torch.empty(0))
            self.raw_s_mid = nn.Parameter(torch.empty(0))
            self.A_mid_real = nn.Parameter(torch.empty(2, 0))
            self.A_mid_imag = nn.Parameter(torch.empty(2, 0))
            self.s_floor_mid = 0.0
            self.max_s_mid = 0.0

        # ----- banque rapide (Hz réels)
        self.Kt_high = Kt_high
        if Kt_high > 0:
            freqs_high = torch.linspace(fmin_high_hz, fmax_high_hz, Kt_high, dtype=torch.float32)
            omega_high = 2 * math.pi * freqs_high  # rad/s
            self.register_buffer("omega_high", omega_high)
            self.raw_s_high = nn.Parameter(torch.full((Kt_high,), -2.0, dtype=torch.float32))
            self.A_high_real = nn.Parameter(torch.randn(2, Kt_high) * 0.01)
            self.A_high_imag = nn.Parameter(torch.randn(2, Kt_high) * 0.01)
            self.s_floor_high = float(s_floor_high)
            self.max_s_high = float(max_s_high)
        else:
            self.register_buffer("omega_high", torch.empty(0))
            self.raw_s_high = nn.Parameter(torch.empty(0))
            self.A_high_real = nn.Parameter(torch.empty(2, 0))
            self.A_high_imag = nn.Parameter(torch.empty(2, 0))
            self.s_floor_high = 0.0
            self.max_s_high = 0.0

        # offset global
        self.base_xy = nn.Parameter(torch.zeros(2, dtype=torch.float32))

    # banque "par durée"
    def _bank_duration(self, t, omega, raw_s, A_real, A_imag, s_floor, max_s):
        if omega.numel() == 0:
            return torch.zeros(t.shape[0], 2, device=t.device)
        T = t.shape[0]
        tt = t.view(T, 1)
        s = s_floor + F.softplus(raw_s)
        s = s.clamp(max=max_s)
        decay = torch.exp(-tt * s.view(1, -1))
        phase = tt * omega.view(1, -1)
        exp_iwt = torch.complex(torch.cos(phase), torch.sin(phase))
        kernel = decay * exp_iwt
        A = torch.complex(A_real, A_imag)
        xy_complex = torch.einsum("tk,dk->td", kernel, A)
        return xy_complex.real

    # banque en Hz réels
    def _bank_hz(self, t, omega_hz, raw_s, A_real, A_imag, s_floor, max_s):
        if omega_hz.numel() == 0:
            return torch.zeros(t.shape[0], 2, device=t.device)
        T = t.shape[0]
        tt = t.view(T, 1)
        s = s_floor + F.softplus(raw_s)
        s = s.clamp(max=max_s)
        decay = torch.exp(-tt * s.view(1, -1))
        phase = tt * omega_hz.view(1, -1)  # t en secondes
        exp_iwt = torch.complex(torch.cos(phase), torch.sin(phase))
        kernel = decay * exp_iwt
        A = torch.complex(A_real, A_imag)
        xy_complex = torch.einsum("tk,dk->td", kernel, A)
        return xy_complex.real

    def forward(self, t_grid: torch.Tensor) -> torch.Tensor:
        t_grid = t_grid.float()

        out = 0.0
        # low (durée)
        out = out + self._bank_duration(
            t_grid,
            self.omega_low,
            self.raw_s_low,
            self.A_low_real,
            self.A_low_imag,
            self.s_floor_low,
            self.max_s_low,
        )
        # mid (Hz)
        out = out + self._bank_hz(
            t_grid,
            self.omega_mid,
            self.raw_s_mid,
            self.A_mid_real,
            self.A_mid_imag,
            self.s_floor_mid,
            self.max_s_mid,
        )
        # high (Hz)
        out = out + self._bank_hz(
            t_grid,
            self.omega_high,
            self.raw_s_high,
            self.A_high_real,
            self.A_high_imag,
            self.s_floor_high,
            self.max_s_high,
        )

        # offset
        out = out + self.base_xy.view(1, 2)
        return out


# ============================================================
# 2. Enregistrement souris brut
# ============================================================
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


# ============================================================
# 3. Préparation
# ============================================================
def prepare_data(ts, xs, ys):
    assert len(ts) == len(xs) == len(ys)
    t_max = float(ts.max())
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    bounds = (xmin, xmax, ymin, ymax)
    t_torch = torch.from_numpy(ts).float()
    xy_torch = torch.from_numpy(np.stack([xs, ys], axis=-1)).float()
    return t_torch, xy_torch, t_max, bounds


# ============================================================
# 4. Pénalité
# ============================================================
def out_of_bounds_penalty(xy_pred, x_min, x_max, y_min, y_max):
    under_x = (x_min - xy_pred[:, 0]).clamp(min=0.0)
    over_x = (xy_pred[:, 0] - x_max).clamp(min=0.0)
    under_y = (y_min - xy_pred[:, 1]).clamp(min=0.0)
    over_y = (xy_pred[:, 1] - y_max).clamp(min=0.0)
    pen = under_x**2 + over_x**2 + under_y**2 + over_y**2
    return pen.mean()


# ============================================================
# 5. Entraînement
# ============================================================
def train_laplace(
    model,
    t_rec,
    xy_rec,
    bounds,
    n_epochs=2000,
    lr=2e-3,
    smooth_w=0.0,
    amp_w=0.001,
    env_w=1.0,
    device="cpu",
):
    model.to(device)
    t_rec = t_rec.to(device)
    xy_rec = xy_rec.to(device)

    x_min, x_max, y_min, y_max = bounds
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(1, n_epochs + 1):
        opt.zero_grad()
        xy_pred = model(t_rec)  # [N,2]

        loss_fit = F.mse_loss(xy_pred, xy_rec)

        if xy_pred.shape[0] > 2 and smooth_w > 0.0:
            dxy = xy_pred[1:] - xy_pred[:-1]
            loss_smooth = (dxy ** 2).mean()
        else:
            loss_smooth = 0.0

        loss_oob = out_of_bounds_penalty(xy_pred, x_min, x_max, y_min, y_max)
        loss_amp = (xy_pred ** 2).mean()

        loss = loss_fit + smooth_w * loss_smooth + env_w * loss_oob + amp_w * loss_amp
        loss.backward()
        opt.step()

        if ep % max(1, n_epochs // 20) == 0 or ep == 1:
            print(
                f"[train] epoch {ep:4d} | loss={loss.item():.6f} "
                f"| fit={loss_fit.item():.6f} "
                f"| oob={loss_oob.item():.6f} "
                f"| smooth={float(loss_smooth):.6f}"
            )

    return model


# ============================================================
# 6. Replay (3 graphes)
# ============================================================
def replay_raw_times(ts, xs, ys, model, bounds, extrap_pct=None, fps=50):
    x_min, x_max, y_min, y_max = bounds

    # 1) enregistré
    xy_rec = np.stack([xs, ys], axis=-1)
    N_rec = xy_rec.shape[0]

    # 2) modèle sur temps bruts
    t_torch_raw = torch.from_numpy(ts).float()
    model.eval()
    with torch.no_grad():
        xy_pred_raw = model(t_torch_raw).cpu().numpy()
    N_raw = xy_pred_raw.shape[0]

    # 3) modèle sur temps EXTRAPOLÉS OU PAS
    if extrap_pct is None or extrap_pct <= 0.0:
        # → pas d'extrapolation : on prend juste les mêmes temps
        ts_ext = ts
        t_torch_ext = torch.from_numpy(ts_ext).float()
        with torch.no_grad():
            xy_pred_ext = model(t_torch_ext).cpu().numpy()
        N_ext = xy_pred_ext.shape[0]
        extrap_title = "Laplace (no extrap) — 3 banks"
    else:
        # → extrapolation demandée
        t0 = float(ts[0])
        t1 = float(ts.max())
        duration = t1 - t0
        t_end_ext = t1 + extrap_pct * duration
        N_ext = int(N_rec * (1.0 + extrap_pct))
        N_ext = max(N_ext, N_rec + 1)
        ts_ext = np.linspace(t0, t_end_ext, N_ext, dtype=np.float32)
        t_torch_ext = torch.from_numpy(ts_ext).float()
        with torch.no_grad():
            xy_pred_ext = model(t_torch_ext).cpu().numpy()
        extrap_title = f"Laplace (extrap. {int(extrap_pct*100)}%) — 3 banks"

    # clamp pour 1 et 2 (pas pour 3)
    xy_pred_raw[:, 0] = np.clip(xy_pred_raw[:, 0], x_min, x_max)
    xy_pred_raw[:, 1] = np.clip(xy_pred_raw[:, 1], y_min, y_max)

    # fig
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5))

    # titre global
    if extrap_pct is None or extrap_pct <= 0.0:
        fig.suptitle("Recorded vs Laplace (ts bruts) vs Laplace (no extrap)")
    else:
        fig.suptitle(f"Recorded vs Laplace (ts bruts) vs Laplace (extrap. {int(extrap_pct*100)}%)")

    # axes 1 & 2 : bornés
    for ax in (ax1, ax2):
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

    # axe 3 :
    if extrap_pct is None or extrap_pct <= 0.0:
        # même cadre que l'enregistrement
        ax3.set_xlim(x_min, x_max)
        ax3.set_ylim(y_min, y_max)
    else:
        # auto-range
        x3_min = min(x_min, float(xy_pred_ext[:, 0].min()))
        x3_max = max(x_max, float(xy_pred_ext[:, 0].max()))
        y3_min = min(y_min, float(xy_pred_ext[:, 1].min()))
        y3_max = max(y_max, float(xy_pred_ext[:, 1].max()))
        pad_x = (x3_max - x3_min) * 0.05 if x3_max > x3_min else 0.05
        pad_y = (y3_max - y3_min) * 0.05 if y3_max > y3_min else 0.05
        ax3.set_xlim(x3_min - pad_x, x3_max + pad_x)
        ax3.set_ylim(y3_min - pad_y, y3_max + pad_y)

    ax3.set_aspect("equal")
    ax3.grid(True, alpha=0.2)

    ax1.set_title("Recorded")
    ax2.set_title("Laplace (ts bruts)")
    ax3.set_title(extrap_title)

    (p1,) = ax1.plot([], [], "o", color="tab:blue")
    (p2,) = ax2.plot([], [], "o", color="tab:red")
    (p3,) = ax3.plot([], [], "o", color="tab:green")

    trail1, = ax1.plot([], [], "-", color="tab:blue", alpha=0.6)
    trail2, = ax2.plot([], [], "-", color="tab:red", alpha=0.6)
    trail3, = ax3.plot([], [], "-", color="tab:green", alpha=0.6)

    ANIM = [None]
    max_frames = max(N_rec, N_raw, xy_pred_ext.shape[0])

    def init():
        return p1, p2, p3, trail1, trail2, trail3

    def update(frame):
        # 1
        if frame < N_rec:
            p1.set_data([xy_rec[frame, 0]], [xy_rec[frame, 1]])
            trail1.set_data(xy_rec[: frame + 1, 0], xy_rec[: frame + 1, 1])
        else:
            p1.set_data([xy_rec[-1, 0]], [xy_rec[-1, 1]])
            trail1.set_data(xy_rec[:, 0], xy_rec[:, 1])

        # 2
        if frame < N_raw:
            p2.set_data([xy_pred_raw[frame, 0]], [xy_pred_raw[frame, 1]])
            trail2.set_data(xy_pred_raw[: frame + 1, 0], xy_pred_raw[: frame + 1, 1])
        else:
            p2.set_data([xy_pred_raw[-1, 0]], [xy_pred_raw[-1, 1]])
            trail2.set_data(xy_pred_raw[:, 0], xy_pred_raw[:, 1])

        # 3
        if frame < xy_pred_ext.shape[0]:
            p3.set_data([xy_pred_ext[frame, 0]], [xy_pred_ext[frame, 1]])
            trail3.set_data(xy_pred_ext[: frame + 1, 0], xy_pred_ext[: frame + 1, 1])
        else:
            p3.set_data([xy_pred_ext[-1, 0]], [xy_pred_ext[-1, 1]])
            trail3.set_data(xy_pred_ext[:, 0], xy_pred_ext[:, 1])

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


# ============================================================
# 7. Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-sec", type=float, default=8.0)

    # banque lente
    parser.add_argument("--Kt-low", type=int, default=32)
    parser.add_argument("--fmin-low", type=float, default=0.2)
    parser.add_argument("--fmax-low", type=float, default=4.0)
    parser.add_argument("--s-floor-low", type=float, default=0.05)
    parser.add_argument("--max-s-low", type=float, default=2.0)

    # banque moyenne
    parser.add_argument("--Kt-mid", type=int, default=48)
    parser.add_argument("--fmin-mid-hz", type=float, default=0.5)
    parser.add_argument("--fmax-mid-hz", type=float, default=6.0)
    parser.add_argument("--s-floor-mid", type=float, default=0.02)
    parser.add_argument("--max-s-mid", type=float, default=1.5)

    # banque rapide
    parser.add_argument("--Kt-high", type=int, default=96)
    parser.add_argument("--fmin-high-hz", type=float, default=6.0)
    parser.add_argument("--fmax-high-hz", type=float, default=40.0)
    parser.add_argument("--s-floor-high", type=float, default=0.0)
    parser.add_argument("--max-s-high", type=float, default=1.0)

    # training
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--smooth-w", type=float, default=0.0)
    parser.add_argument("--amp-w", type=float, default=0.001)
    parser.add_argument("--env-w", type=float, default=1.0)

    # extrapolation (facultatif)
    parser.add_argument("--extrap-pct", type=float, default=None,
                        help="si absent ou <=0 → pas d'extrapolation")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    print(f">>> STEP 1: record {args.record_sec}s …")
    ts, xs, ys = record_mouse(duration=args.record_sec)
    print(f"Recorded {len(ts)} samples.")

    print(">>> STEP 2: prepare …")
    t_rec, xy_rec, t_max, bounds = prepare_data(ts, xs, ys)
    print(f"Duration recorded: {t_max:.2f}s, bounds={bounds}")

    print(">>> STEP 3: build model …")
    model = LaplaceMotion3Banks(
        duration=t_max,
        Kt_low=args.Kt_low,
        fmin_low=args.fmin_low,
        fmax_low=args.fmax_low,
        s_floor_low=args.s_floor_low,
        max_s_low=args.max_s_low,
        Kt_mid=args.Kt_mid,
        fmin_mid_hz=args.fmin_mid_hz,
        fmax_mid_hz=args.fmax_mid_hz,
        s_floor_mid=args.s_floor_mid,
        max_s_mid=args.max_s_mid,
        Kt_high=args.Kt_high,
        fmin_high_hz=args.fmin_high_hz,
        fmax_high_hz=args.fmax_high_hz,
        s_floor_high=args.s_floor_high,
        max_s_high=args.max_s_high,
    ).to(device)

    print(">>> STEP 4: train …")
    model = train_laplace(
        model,
        t_rec,
        xy_rec,
        bounds,
        n_epochs=args.epochs,
        lr=args.lr,
        smooth_w=args.smooth_w,
        amp_w=args.amp_w,
        env_w=args.env_w,
        device=device,
    )

    print(">>> STEP 5: replay …")
    replay_raw_times(
        ts,
        xs,
        ys,
        model,
        bounds,
        extrap_pct=args.extrap_pct,
        fps=60,
    )


if __name__ == "__main__":
    main()
