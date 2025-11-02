#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interactive Laplace Spatio-Temporal Demo — 3 banks (auto ratio HF réduit)
+ correction des segments immobiles
--------------------------------------------------------------------------

commande : 
python 22-laplace-spatio-temporal-reproduce-3banks-auto.py --record-sec 30  --fmax-high-hz 0.0000000000005 --epochs 4000 --smooth-w 0 --stay-thresh 0.000001 --stay-w 150

Problème corrigé :
- quand l'utilisateur reste (presque) au même endroit pendant l'enregistrement,
  le modèle, à la repro, faisait parfois un zigzag ou un petit aller-retour,
  parce qu'il devait passer plusieurs fois sur la même coordonnée à des temps différents.
- on détecte ces pas "immobiles" dans les données brutes
- on ajoute UNIQUEMENT sur ces pas-là une pénalité de vitesse pour le modèle

C'est une contrainte "data-consistante" : on ne lisse pas tout, on ne touche
que les endroits où toi-même tu n'as pas bougé.
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
# 0. Heuristiques auto (N, durée) -> (Kt_low, Kt_mid, Kt_high)
# ============================================================
def auto_banks(n_samples: int, duration: float):
    """
    Plus la démo est longue, plus on RÉDUIT la banque haute.
    (version que tu utilisais déjà)
    """
 
    k = int(round(n_samples / 10))
    k = max(32, int(4 * k))
    k_low = k
    k_mid = max(32, int(1.0 * k))
    # on écrase vraiment la HF
    k_high = max(16, int(0.6 * k))
    return k_low, k_mid, k_high


def estimate_target_loss(n_samples: int) -> float:
    base = 2.5e-4
    extra = max(0, n_samples - 300) * 3.5e-6
    return base + extra


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

    def _bank_hz(self, t, omega_hz, raw_s, A_real, A_imag, s_floor, max_s):
        if omega_hz.numel() == 0:
            return torch.zeros(t.shape[0], 2, device=t.device)
        T = t.shape[0]
        tt = t.view(T, 1)
        s = s_floor + F.softplus(raw_s)
        s = s.clamp(max=max_s)
        decay = torch.exp(-tt * s.view(1, -1))
        phase = tt * omega_hz.view(1, -1)
        exp_iwt = torch.complex(torch.cos(phase), torch.sin(phase))
        kernel = decay * exp_iwt
        A = torch.complex(A_real, A_imag)
        xy_complex = torch.einsum("tk,dk->td", kernel, A)
        return xy_complex.real

    def forward(self, t_grid: torch.Tensor) -> torch.Tensor:
        t_grid = t_grid.float()
        out = 0.0
        out = out + self._bank_duration(
            t_grid,
            self.omega_low,
            self.raw_s_low,
            self.A_low_real,
            self.A_low_imag,
            self.s_floor_low,
            self.max_s_low,
        )
        out = out + self._bank_hz(
            t_grid,
            self.omega_mid,
            self.raw_s_mid,
            self.A_mid_real,
            self.A_mid_imag,
            self.s_floor_mid,
            self.max_s_mid,
        )
        out = out + self._bank_hz(
            t_grid,
            self.omega_high,
            self.raw_s_high,
            self.A_high_real,
            self.A_high_imag,
            self.s_floor_high,
            self.max_s_high,
        )
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
# 4. Pénalité out of bounds
# ============================================================
def out_of_bounds_penalty(xy_pred, x_min, x_max, y_min, y_max):
    under_x = (x_min - xy_pred[:, 0]).clamp(min=0.0)
    over_x = (xy_pred[:, 0] - x_max).clamp(min=0.0)
    under_y = (y_min - xy_pred[:, 1]).clamp(min=0.0)
    over_y = (xy_pred[:, 1] - y_max).clamp(min=0.0)
    pen = under_x**2 + over_x**2 + under_y**2 + over_y**2
    return pen.mean()


# ============================================================
# 4bis. Détection des pas immobiles
# ============================================================
def build_stationary_mask(xy_rec: torch.Tensor, stay_thresh: float = 1e-3):
    """
    xy_rec: [N,2]
    retourne un masque booléen de taille N-1 (pour les différences)
    True là où le VRAI mouvement est quasi nul → on veut vitesse modèle ~ 0 aussi
    """
    if xy_rec.shape[0] < 2:
        return torch.zeros(0, dtype=torch.bool)
    dxy = xy_rec[1:] - xy_rec[:-1]
    dist = torch.linalg.norm(dxy, dim=-1)
    mask = dist < stay_thresh
    return mask  # [N-1]


# ============================================================
# 5. Entraînement avec pénalité locale sur segments immobiles
# ============================================================
def train_laplace(
    model,
    t_rec,
    xy_rec,
    bounds,
    n_epochs=2000,
    lr=2e-3,
    smooth_w=0.0,
    amp_w=0.0,
    env_w=1.0,
    stay_thresh=1e-3,
    stay_w=8.0,
    device="cpu",
):
    model.to(device)
    t_rec = t_rec.to(device)
    xy_rec = xy_rec.to(device)

    x_min, x_max, y_min, y_max = bounds
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # masque des endroits où TU n'as pas bougé
    stay_mask = build_stationary_mask(xy_rec, stay_thresh=stay_thresh).to(device)

    last_loss = None

    for ep in range(1, n_epochs + 1):
        opt.zero_grad()
        xy_pred = model(t_rec)  # [N,2]

        loss_fit = F.mse_loss(xy_pred, xy_rec)

        # pénalité hors cadre
        loss_oob = out_of_bounds_penalty(xy_pred, x_min, x_max, y_min, y_max)

        # pénalité sur segments immobiles
        if xy_pred.shape[0] > 1 and stay_mask.numel() == xy_pred.shape[0] - 1:
            dxy_pred = xy_pred[1:] - xy_pred[:-1]
            # on ne pénalise QUE là où toi tu étais immobile
            if stay_mask.any():
                loss_stay = (dxy_pred[stay_mask] ** 2).mean()
            else:
                loss_stay = torch.tensor(0.0, device=device)
        else:
            loss_stay = torch.tensor(0.0, device=device)

        # lissage global optionnel (tu le gardes à 0)
        if xy_pred.shape[0] > 2 and smooth_w > 0.0:
            dxy = xy_pred[1:] - xy_pred[:-1]
            loss_smooth = (dxy ** 2).mean()
        else:
            loss_smooth = torch.tensor(0.0, device=device)

        loss_amp = (xy_pred ** 2).mean() if amp_w > 0 else torch.tensor(0.0, device=device)

        loss = (
            loss_fit
            + env_w * loss_oob
            + smooth_w * loss_smooth
            + amp_w * loss_amp
            + stay_w * loss_stay
        )
        loss.backward()
        opt.step()

        last_loss = loss.item()

        if ep % max(1, n_epochs // 20) == 0 or ep == 1:
            print(
                f"[train] epoch {ep:4d} "
                f"| loss={loss.item():.6f} "
                f"| fit={loss_fit.item():.6f} "
                f"| oob={loss_oob.item():.6f} "
                f"| stay={loss_stay.item():.6f}"
            )

    return model, last_loss


# ============================================================
# 6. Replay (3 graphes)
# ============================================================
def replay_raw_times(ts, xs, ys, model, bounds, extrap_pct=None, fps=50):
    x_min, x_max, y_min, y_max = bounds

    xy_rec = np.stack([xs, ys], axis=-1)
    N_rec = xy_rec.shape[0]

    t_torch_raw = torch.from_numpy(ts).float()
    model.eval()
    with torch.no_grad():
        xy_pred_raw = model(t_torch_raw).cpu().numpy()
    N_raw = xy_pred_raw.shape[0]

    if extrap_pct is None or extrap_pct <= 0.0:
        ts_ext = ts
        with torch.no_grad():
            xy_pred_ext = model(torch.from_numpy(ts_ext).float()).cpu().numpy()
        extrap_title = "Laplace (no extrap) — 3 banks"
    else:
        t0 = float(ts[0])
        t1 = float(ts.max())
        duration = t1 - t0
        t_end_ext = t1 + extrap_pct * duration
        N_ext = int(N_rec * (1.0 + extrap_pct))
        N_ext = max(N_ext, N_rec + 1)
        ts_ext = np.linspace(t0, t_end_ext, N_ext, dtype=np.float32)
        with torch.no_grad():
            xy_pred_ext = model(torch.from_numpy(ts_ext).float()).cpu().numpy()
        extrap_title = f"Laplace (extrap. {int(extrap_pct*100)}%) — 3 banks"

    # clamp pour affichage 1 et 2
    xy_pred_raw[:, 0] = np.clip(xy_pred_raw[:, 0], x_min, x_max)
    xy_pred_raw[:, 1] = np.clip(xy_pred_raw[:, 1], y_min, y_max)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5))

    if extrap_pct is None or extrap_pct <= 0.0:
        fig.suptitle("Recorded vs Laplace (ts bruts) vs Laplace (no extrap)")
    else:
        fig.suptitle(f"Recorded vs Laplace (ts bruts) vs Laplace (extrap. {int(extrap_pct*100)}%)")

    for ax in (ax1, ax2):
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

    if extrap_pct is None or extrap_pct <= 0.0:
        ax3.set_xlim(x_min, x_max)
        ax3.set_ylim(y_min, y_max)
    else:
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

    # si on met -1 → auto
    parser.add_argument("--Kt-low", type=int, default=-1)
    parser.add_argument("--Kt-mid", type=int, default=-1)
    parser.add_argument("--Kt-high", type=int, default=-1)

    parser.add_argument("--fmin-low", type=float, default=0.2)
    parser.add_argument("--fmax-low", type=float, default=4.0)
    parser.add_argument("--s-floor-low", type=float, default=0.05)
    parser.add_argument("--max-s-low", type=float, default=2.0)

    parser.add_argument("--fmin-mid-hz", type=float, default=0.5)
    parser.add_argument("--fmax-mid-hz", type=float, default=6.0)
    parser.add_argument("--s-floor-mid", type=float, default=0.02)
    parser.add_argument("--max-s-mid", type=float, default=1.5)

    parser.add_argument("--fmin-high-hz", type=float, default=6.0)
    parser.add_argument("--fmax-high-hz", type=float, default=40.0)
    parser.add_argument("--s-floor-high", type=float, default=0.0)
    parser.add_argument("--max-s-high", type=float, default=1.0)

    # training
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--smooth-w", type=float, default=0.0)
    parser.add_argument("--amp-w", type=float, default=0.0)
    parser.add_argument("--env-w", type=float, default=1.0)

    # correction immobiles
    parser.add_argument("--stay-thresh", type=float, default=1e-3,
                        help="mouvement réel en-dessous duquel on considère que tu étais immobile")
    parser.add_argument("--stay-w", type=float, default=8.0,
                        help="poids de la pénalité de vitesse du modèle sur les segments immobiles")

    # extrapolation optionnelle
    parser.add_argument("--extrap-pct", type=float, default=None)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    print(f">>> STEP 1: record {args.record_sec}s …")
    ts, xs, ys = record_mouse(duration=args.record_sec)
    N = len(ts)
    print(f"Recorded {N} samples.")

    print(">>> STEP 2: prepare …")
    t_rec, xy_rec, t_max, bounds = prepare_data(ts, xs, ys)
    print(f"Duration recorded: {t_max:.2f}s, bounds={bounds}")

    # --- AUTO BANKS ?
    if args.Kt_low < 0 or args.Kt_mid < 0 or args.Kt_high < 0:
        k_low, k_mid, k_high = auto_banks(N, t_max)
        print(f"[auto] using Kt-low={k_low}, Kt-mid={k_mid}, Kt-high={k_high}")
    else:
        k_low, k_mid, k_high = args.Kt_low, args.Kt_mid, args.Kt_high
        print(f"[manual] using Kt-low={k_low}, Kt-mid={k_mid}, Kt-high={k_high}")

    target_loss = estimate_target_loss(N)
    print(f"[auto] estimated good loss ≈ {target_loss:.6f}")

    print(">>> STEP 3: build model …")
    model = LaplaceMotion3Banks(
        duration=t_max,
        Kt_low=k_low,
        fmin_low=args.fmin_low,
        fmax_low=args.fmax_low,
        s_floor_low=args.s_floor_low,
        max_s_low=args.max_s_low,
        Kt_mid=k_mid,
        fmin_mid_hz=args.fmin_mid_hz,
        fmax_mid_hz=args.fmax_mid_hz,
        s_floor_mid=args.s_floor_mid,
        max_s_mid=args.max_s_mid,
        Kt_high=k_high,
        fmin_high_hz=args.fmin_high_hz,
        fmax_high_hz=args.fmax_high_hz,
        s_floor_high=args.s_floor_high,
        max_s_high=args.max_s_high,
    ).to(device)

    print(">>> STEP 4: train …")
    model, final_loss = train_laplace(
        model,
        t_rec,
        xy_rec,
        bounds,
        n_epochs=args.epochs,
        lr=args.lr,
        smooth_w=args.smooth_w,
        amp_w=args.amp_w,
        env_w=args.env_w,
        stay_thresh=args.stay_thresh,
        stay_w=args.stay_w,
        device=device,
    )

    print(f"[eval] final loss = {final_loss:.6f}")
    if final_loss <= target_loss * 1.25:
        print("[eval] ✅ loss dans la plage attendue → banques OK")
    else:
        print("[eval] ⚠️ loss plus haute que prévu → essaye d'augmenter k_mid seulement (+20)")

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
