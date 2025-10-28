#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Laplace Drawing v3 - Approche simplifiée et robuste
----------------------------------------------------
Changements majeurs:
- Architecture plus simple (sin/cos direct sans complexité inutile)
- Initialisation par régression linéaire sur FFT
- Poids équilibrés dès le départ
- Pas de time-warp, pas de Kuramoto
- Focus sur la position d'abord, puis affinage

Usage:
python laplace_drawing_face_v3.py --png face.png
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import trange

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# ============================================================
# 1) Extraction du contour
# ============================================================

def load_and_extract_polyline(
    png_path,
    target_points=4000,
    canny1=100, canny2=200,
    invert_y=True,
    morph_iterations=1,
):
    """Extrait et normalise un contour fermé."""
    try:
        import cv2
    except ImportError:
        raise RuntimeError("pip install opencv-python")

    img = Image.open(png_path).convert("L")
    im = np.array(img)

    im_blur = cv2.GaussianBlur(im, (5, 5), 0)
    edges = cv2.Canny(im_blur, canny1, canny2)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=morph_iterations)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError("Aucun contour trouvé.")
    curves = [c[:, 0, :].astype(np.float32) for c in contours if len(c) >= 10]

    def stitch(curves_list):
        curves_list = sorted(curves_list, key=lambda a: len(a), reverse=True)
        path = curves_list.pop(0)
        def endpoints(seg): return seg[0], seg[-1]
        while curves_list:
            p0, p1 = endpoints(path)
            best_i, best_mode, best_dir, best_d = None, None, None, 1e9
            for i, seg in enumerate(curves_list):
                s0, s1 = endpoints(seg)
                cand = [
                    (np.linalg.norm(p1 - s0), ("append", +1)),
                    (np.linalg.norm(p1 - s1), ("append", -1)),
                    (np.linalg.norm(p0 - s0), ("prepend", -1)),
                    (np.linalg.norm(p0 - s1), ("prepend", +1)),
                ]
                dmin, how = min(cand, key=lambda x: x[0])
                if dmin < best_d:
                    best_d, best_i, (best_mode, best_dir) = dmin, i, how
            seg = curves_list.pop(best_i)
            if best_dir < 0: seg = seg[::-1].copy()
            path = np.vstack([path, seg]) if best_mode == "append" else np.vstack([seg, path])
        return path

    stitched = stitch(curves)
    if np.linalg.norm(stitched[0] - stitched[-1]) > 1.0:
        stitched = np.vstack([stitched, stitched[0]])

    dxy = np.diff(stitched, axis=0)
    seg = np.sqrt((dxy**2).sum(1))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total <= 0:
        raise RuntimeError("Contour dégénéré.")
    stitched_wrap = np.vstack([stitched, stitched[1]])
    s_wrap = np.concatenate([s, [total + seg[0]]])

    s_uniform = np.linspace(0, total, target_points, endpoint=False)
    x = np.interp(s_uniform, s_wrap, stitched_wrap[:, 0])
    y = np.interp(s_uniform, s_wrap, stitched_wrap[:, 1])
    poly = np.stack([x, y], axis=1)

    minxy = poly.min(axis=0)
    maxxy = poly.max(axis=0)
    center = (minxy + maxxy) / 2.0
    scale = (maxxy - minxy).max() / 2.0 + 1e-9
    poly_norm = (poly - center) / scale

    if invert_y:
        poly_norm[:, 1] *= -1.0

    return poly_norm.astype(np.float32)


# ============================================================
# 2) Modèle simplifié
# ============================================================

class SimpleFourierModel(nn.Module):
    """
    Modèle de Fourier simple et direct:
    X(t) = dc_x + Σ [a_xk * sin(ωk*t) + b_xk * cos(ωk*t)]
    Y(t) = dc_y + Σ [a_yk * sin(ωk*t) + b_yk * cos(ωk*t)]
    """
    def __init__(self, n_units=200, period=10.0):
        super().__init__()
        self.n_units = n_units
        self.period = period
        
        # Fréquences harmoniques fixes (non apprises)
        k = torch.arange(1, n_units + 1, dtype=torch.float32)
        self.register_buffer('omega', 2 * math.pi * k / period)
        
        # Coefficients de Fourier (X et Y séparés)
        self.a_x = nn.Parameter(torch.zeros(n_units))
        self.b_x = nn.Parameter(torch.zeros(n_units))
        self.a_y = nn.Parameter(torch.zeros(n_units))
        self.b_y = nn.Parameter(torch.zeros(n_units))
        
        # Offset DC
        self.dc = nn.Parameter(torch.zeros(2))
        
        # Décalage temporel
        self.tau = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, t):
        """
        t: [B, T] temps
        retourne: [B, T, 2] positions
        """
        t_shifted = t + self.tau  # [B, T]
        
        # Phase: [N] × [B, T] -> [B, T, N]
        phase = self.omega.unsqueeze(0).unsqueeze(0) * t_shifted.unsqueeze(2)
        
        sin_phase = torch.sin(phase)  # [B, T, N]
        cos_phase = torch.cos(phase)  # [B, T, N]
        
        # X(t)
        x = (self.a_x * sin_phase + self.b_x * cos_phase).sum(dim=2) + self.dc[0]
        
        # Y(t)
        y = (self.a_y * sin_phase + self.b_y * cos_phase).sum(dim=2) + self.dc[1]
        
        return torch.stack([x, y], dim=2)  # [B, T, 2]
    
    def init_from_fft(self, signal, t):
        """
        Initialise les coefficients par FFT.
        signal: [T, 2] numpy
        t: [T] numpy
        """
        T = signal.shape[0]
        
        # FFT pour X et Y
        fft_x = np.fft.rfft(signal[:, 0])
        fft_y = np.fft.rfft(signal[:, 1])
        
        # DC component
        self.dc.data[0] = float(fft_x[0].real / T)
        self.dc.data[1] = float(fft_y[0].real / T)
        
        # Coefficients harmoniques
        n = min(self.n_units, len(fft_x) - 1)
        for k in range(n):
            idx = k + 1  # Skip DC
            
            # X: FFT[k] = (a - ib) * T/2
            # => a = -2*imag/T, b = 2*real/T
            self.a_x.data[k] = float(-2 * fft_x[idx].imag / T)
            self.b_x.data[k] = float(2 * fft_x[idx].real / T)
            
            self.a_y.data[k] = float(-2 * fft_y[idx].imag / T)
            self.b_y.data[k] = float(2 * fft_y[idx].real / T)


# ============================================================
# 3) Dérivées
# ============================================================

def compute_derivatives(x, dt):
    """
    x: [B, T, C]
    retourne: vel [B, T, C], acc [B, T, C]
    """
    B, T, C = x.shape
    x_p = x.permute(0, 2, 1)  # [B, C, T]
    
    # Gradient central
    kernel = torch.tensor([[-0.5, 0.0, 0.5]], dtype=x.dtype, device=x.device) / dt
    kernel = kernel.view(1, 1, 3).expand(C, 1, 3).contiguous()
    
    v_p = F.conv1d(x_p, kernel, padding=1, groups=C)
    vel = v_p.permute(0, 2, 1)
    
    a_p = F.conv1d(v_p, kernel, padding=1, groups=C)
    acc = a_p.permute(0, 2, 1)
    
    return vel, acc


# ============================================================
# 4) Entraînement
# ============================================================

def train_model(
    poly,
    duration=10.0,
    n_units=200,
    epochs=6000,
    lr=5e-4,
    device='cuda'
):
    """Entraîne le modèle avec une stratégie progressive."""
    
    T = poly.shape[0]
    t = np.linspace(0, duration, T, endpoint=False).astype(np.float32)
    dt = duration / T
    
    t_t = torch.from_numpy(t).unsqueeze(0).to(device)
    target = torch.from_numpy(poly).unsqueeze(0).to(device)
    
    # Modèle
    model = SimpleFourierModel(n_units=n_units, period=duration).to(device)
    
    # Initialisation FFT
    print("   Initialisation par FFT...")
    model.init_from_fft(poly, t)
    
    # Vérification post-init
    with torch.no_grad():
        pred_init = model(t_t)
        mse_init = F.mse_loss(pred_init, target).item()
        print(f"   MSE après init FFT: {mse_init:.6f}")
    
    # Optimizer avec learning rate adaptative
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=300, min_lr=lr*0.01
    )
    
    losses = []
    best_loss = float('inf')
    best_state = None
    patience_counter = 0
    max_patience = 800
    
    # PHASE UNIQUE avec pondération progressive
    print(f"\n   Entraînement progressif...")
    
    for ep in trange(epochs, desc="Training"):
        optimizer.zero_grad()
        
        pred = model(t_t)
        
        # Pondération progressive (augmente doucement vel/acc/spec)
        progress = min(1.0, ep / (epochs * 0.7))
        
        # Loss position (toujours importante)
        loss_pos = F.mse_loss(pred, target)
        
        # Clôture renforcée
        closure_pos = F.mse_loss(pred[:, 0, :], pred[:, -1, :])
        # Clôture sur région (moyenne des 5 premiers/derniers points)
        closure_region = F.mse_loss(
            pred[:, :5, :].mean(dim=1),
            pred[:, -5:, :].mean(dim=1)
        )
        
        loss = loss_pos + 2.0 * closure_pos + 1.0 * closure_region
        
        # Ajout progressif des termes de régularisation
        if progress > 0.3:
            vel_pred, acc_pred = compute_derivatives(pred, dt)
            vel_tgt, acc_tgt = compute_derivatives(target, dt)
            
            # Poids très faibles pour éviter l'explosion
            w_vel = 0.02 * progress
            w_acc = 0.0005 * progress
            
            loss_vel = F.mse_loss(vel_pred, vel_tgt) * w_vel
            loss_acc = F.mse_loss(acc_pred, acc_tgt) * w_acc
            
            # Clôture vitesse
            closure_vel = F.mse_loss(vel_pred[:, 0, :], vel_pred[:, -1, :]) * w_vel
            
            loss = loss + loss_vel + loss_acc + closure_vel
        
        # Perte spectrale (douce)
        if progress > 0.5:
            X_pred = torch.fft.rfft(pred, dim=1)
            X_tgt = torch.fft.rfft(target, dim=1)
            loss_spec = F.l1_loss(X_pred.abs(), X_tgt.abs()) * (0.1 * progress)
            loss = loss + loss_spec
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        loss_val = loss.item()
        losses.append(loss_val)
        scheduler.step(loss_val)
        
        # Sauvegarde du meilleur modèle
        if loss_val < best_loss:
            best_loss = loss_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter > max_patience:
            print(f"\n   Early stopping à epoch {ep+1}")
            break
        
        if (ep + 1) % 500 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\n   [{ep+1}/{epochs}] loss={loss_val:.6f} "
                  f"pos={loss_pos.item():.6f} "
                  f"closure_pos={closure_pos.item():.6f} "
                  f"lr={current_lr:.2e}")
    
    # Restaurer le meilleur modèle
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"\n   Meilleure loss: {best_loss:.6f}")
    
    # Prédiction finale
    with torch.no_grad():
        pred_final = model(t_t).cpu().numpy()[0]
    
    return pred_final, t, losses, model


# ============================================================
# 5) Exports
# ============================================================

def save_svg(points, path="laplace_drawing.svg", size=(800, 800)):
    """Sauvegarde en SVG."""
    W, H = size
    pts = (points + 1.0) / 2.0
    x = pts[:, 0] * W
    y = (1.0 - pts[:, 1]) * H
    
    d = f"M {x[0]:.2f},{y[0]:.2f} "
    d += " ".join(f"L {xi:.2f},{yi:.2f}" for xi, yi in zip(x[1:], y[1:]))
    d += " Z"
    
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">
  <path d="{d}" fill="none" stroke="black" stroke-width="2"/>
</svg>'''
    
    with open(path, 'w') as f:
        f.write(svg)


def save_animation(poly, pred, path="laplace_drawing.mp4", fps=60, duration=10.0):
    """Crée une animation de la reconstruction progressive."""
    try:
        from matplotlib.animation import FFMpegWriter, FuncAnimation
    except ImportError:
        print("   ⚠️ matplotlib.animation non disponible, skip animation")
        return
    
    T = pred.shape[0]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.set_title("Laplace Drawing - Reconstruction", fontsize=16, pad=20)
    ax.axis('off')
    
    # Lignes
    line_target, = ax.plot([], [], 'gray', lw=2, alpha=0.3, label='Target')
    line_model, = ax.plot([], [], 'red', lw=3, label='Model')
    
    # Point courant
    point_current, = ax.plot([], [], 'ro', markersize=8)
    
    # Texte progression
    text_progress = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                           fontsize=12, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.legend(loc='upper right', fontsize=12)
    
    def init():
        line_target.set_data([], [])
        line_model.set_data([], [])
        point_current.set_data([], [])
        text_progress.set_text('')
        return line_target, line_model, point_current, text_progress
    
    def update(frame):
        # Progression non-linéaire (plus rapide au début)
        progress = (frame / T) ** 0.7
        idx = int(progress * T)
        idx = min(idx, T - 1)
        
        if idx > 0:
            line_target.set_data(poly[:idx, 0], poly[:idx, 1])
            line_model.set_data(pred[:idx, 0], pred[:idx, 1])
            point_current.set_data([pred[idx-1, 0]], [pred[idx-1, 1]])
            
            # Affiche le temps et la progression
            t_current = (idx / T) * duration
            text_progress.set_text(f't = {t_current:.2f}s\n{100*idx/T:.1f}%')
        
        return line_target, line_model, point_current, text_progress
    
    anim = FuncAnimation(fig, update, init_func=init, frames=T,
                        interval=1000/fps, blit=True, repeat=True)
    
    # Sauvegarde
    try:
        writer = FFMpegWriter(fps=fps, bitrate=5000, codec='h264')
        anim.save(path, writer=writer)
        print(f"   → {path}")
    except Exception as e:
        print(f"   ⚠️ Erreur FFMpeg ({e}), tentative GIF...")
        try:
            gif_path = path.replace('.mp4', '.gif')
            anim.save(gif_path, writer='pillow', fps=fps)
            print(f"   → {gif_path}")
        except Exception as e2:
            print(f"   ⚠️ Impossible de créer l'animation: {e2}")
    
    plt.close(fig)


def plot_results(poly, pred, losses):
    """Visualisations."""
    fig, axs = plt.subplots(2, 2, figsize=(14, 14))
    
    # Tracé 2D
    axs[0, 0].plot(poly[:, 0], poly[:, 1], 'gray', lw=2, alpha=0.5, label='Target')
    axs[0, 0].plot(pred[:, 0], pred[:, 1], 'red', lw=2, label='Model')
    axs[0, 0].set_aspect('equal')
    axs[0, 0].set_title('Trajectoire 2D', fontsize=14)
    axs[0, 0].legend()
    axs[0, 0].axis('off')
    
    # Loss
    axs[0, 1].plot(losses, lw=1.5)
    axs[0, 1].set_yscale('log')
    axs[0, 1].set_title('Loss totale', fontsize=14)
    axs[0, 1].set_xlabel('Epochs')
    axs[0, 1].grid(alpha=0.3)
    
    # Erreur ponctuelle
    error = np.linalg.norm(pred - poly, axis=1)
    mae = error.mean()
    axs[1, 0].plot(error, lw=1.5)
    axs[1, 0].set_title(f'Erreur ponctuelle (MAE={mae:.4f})', fontsize=14)
    axs[1, 0].set_xlabel('Point index')
    axs[1, 0].set_ylabel('Distance')
    axs[1, 0].grid(alpha=0.3)
    
    # Comparaison X et Y
    axs[1, 1].plot(poly[:, 0], 'gray', lw=1.5, alpha=0.5, label='Target X')
    axs[1, 1].plot(pred[:, 0], 'red', lw=1.5, alpha=0.8, label='Model X')
    axs[1, 1].plot(poly[:, 1], 'lightblue', lw=1.5, alpha=0.5, label='Target Y')
    axs[1, 1].plot(pred[:, 1], 'blue', lw=1.5, alpha=0.8, label='Model Y')
    axs[1, 1].set_title('Coordonnées X et Y', fontsize=14)
    axs[1, 1].set_xlabel('Point index')
    axs[1, 1].legend()
    axs[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('laplace_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 6) Main
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Laplace Drawing v3 - Simplifié")
    parser.add_argument("--png", type=str, default="face.png")
    parser.add_argument("--points", type=int, default=3000,
                       help="Nombre de points (2000-4000)")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--units", type=int, default=300,
                       help="Nombre d'harmoniques (200-400)")
    parser.add_argument("--epochs", type=int, default=6000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--canny1", type=int, default=100)
    parser.add_argument("--canny2", type=int, default=200)
    
    args = parser.parse_args()
    
    print("="*60)
    print("LAPLACE DRAWING v3 - Approche simplifiée")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # 1) Extraction
    print("\n1) Extraction du contour...")
    poly = load_and_extract_polyline(
        args.png,
        target_points=args.points,
        canny1=args.canny1,
        canny2=args.canny2,
    )
    print(f"   → {len(poly)} points extraits")
    
    # 2) Entraînement
    print("\n2) Entraînement du modèle...")
    pred, t, losses, model = train_model(
        poly,
        duration=args.duration,
        n_units=args.units,
        epochs=args.epochs,
        lr=args.lr,
        device=device
    )
    
    # 3) Métriques
    print("\n3) Métriques finales:")
    mse = np.mean((pred - poly)**2)
    mae = np.mean(np.abs(pred - poly))
    max_err = np.max(np.linalg.norm(pred - poly, axis=1))
    print(f"   MSE:  {mse:.6f}")
    print(f"   MAE:  {mae:.6f}")
    print(f"   Max:  {max_err:.6f}")
    
    # 4) Exports
    print("\n4) Exports...")
    save_svg(pred, "laplace_drawing.svg")
    print("   → laplace_drawing.svg")
    
    save_animation(poly, pred, "laplace_drawing.mp4", fps=60, duration=args.duration)
    
    plot_results(poly, pred, losses)
    print("   → laplace_analysis.png")
    
    print("\n" + "="*60)
    print("Terminé !")
    print("="*60)