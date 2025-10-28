#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Joint-Space Laplace Driver (FK only)
- 6 articulations revolutes
- θ_j(t) = somme laplacienne amortie (harmoniques fixes)
- Entraînement: MSE sur la position de l'effecteur
- Visu: 3D aspect 1:1:1 + animation bras + trajectoire
- Caméra: statique par défaut, option --spin pour tourner

Exemples :
    python 12-laplace_jointspace_fk.py --epochs 1000 --n_units 64 --trajectory circle_flat
    python 12-laplace_jointspace_fk.py --epochs 1200 --n_units 96 --trajectory lemniscate
    python 12-laplace_jointspace_fk.py --trajectory circle_wavy --spin
"""

# ==== backend interactif (macOS) AVANT pyplot ====
import matplotlib
try:
    matplotlib.use("MacOSX")  # ou "Qt5Agg" si PyQt5 dispo
except Exception:
    pass

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

import argparse, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Empêche le GC de tuer l'anim
ANIMS = []

# ----------------------------- Args
p = argparse.ArgumentParser()
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
p.add_argument("--epochs", type=int, default=3000)
p.add_argument("--lr", type=float, default=1e-2)
p.add_argument("--n_units", type=int, default=64, help="nombre d'harmoniques (K)")
p.add_argument("--duration", type=float, default=2.0)
p.add_argument("--n_points", type=int, default=400)
p.add_argument("--trajectory", type=str, default="circle_flat",
               choices=["circle_flat","circle_wavy","lemniscate"])
p.add_argument("--seed", type=int, default=123)
p.add_argument("--spin", action="store_true", help="fait tourner la caméra pendant l'anim")
p.add_argument("--z_penalty", type=float, default=10.0, help="poids de la pénalité sur variations Z (lissage)")
p.add_argument("--z_weight", type=float, default=10.0, help="poids relatif de l'erreur Z vs XY")
args = p.parse_args()

torch.manual_seed(args.seed); np.random.seed(args.seed)
device = torch.device(args.device)
print(f"Device: {device}")

# ----------------------------- Trajectoires cibles (monde)
def generate_trajectory(kind="circle_flat", n=400, duration=2.0):
    t = np.linspace(0, duration, n)
    if kind == "circle_flat":
        R = 60.0
        x = R*np.cos(2*np.pi*t/duration)
        y = R*np.sin(2*np.pi*t/duration)
        z = np.full_like(x, 50.0)  # ajusté pour base à z=0
    elif kind == "circle_wavy":
        R = 60.0
        x = R*np.cos(2*np.pi*t/duration)
        y = R*np.sin(2*np.pi*t/duration)
        z = 50.0 + 10.0*np.sin(4*np.pi*t/duration)  # ajusté pour base à z=0
    else:  # lemniscate (∞) dans XY, Z constant
        a = 60.0
        s = np.linspace(-np.pi/2, 3*np.pi/2, n)
        x = a*np.sin(s)/(1+np.cos(s)**2)
        y = a*np.sin(s)*np.cos(s)/(1+np.cos(s)**2)
        z = np.full_like(x, 50.0)  # ajusté pour base à z=0
    return np.stack([x,y,z], axis=-1), t

target_np, t_np = generate_trajectory(args.trajectory, args.n_points, args.duration)
target = torch.tensor(target_np, dtype=torch.float32, device=device)  # [T,3]
T = target.shape[0]

# Debug: vérifier les données cibles
print(f"📊 Trajectoire cible '{args.trajectory}':")
print(f"   X: min={target_np[:,0].min():.2f}, max={target_np[:,0].max():.2f}, mean={target_np[:,0].mean():.2f}")
print(f"   Y: min={target_np[:,1].min():.2f}, max={target_np[:,1].max():.2f}, mean={target_np[:,1].mean():.2f}")
print(f"   Z: min={target_np[:,2].min():.2f}, max={target_np[:,2].max():.2f}, mean={target_np[:,2].mean():.2f}, std={target_np[:,2].std():.4f}")
print(f"   Nombre de points: {T}")

# ----------------------------- Bras 6-DoF (DH convention standard)
link_a    = [0.0,  30.0, 25.0,  0.0,  0.0,  0.0]   # unités arbitraires
link_alpha= [np.pi/2, 0.0, 0.0, np.pi/2, -np.pi/2, 0.0]
link_d    = [0.0,   0.0,  0.0, 35.0,    0.0,   15.0]  # base à z=0
n_joints = 6

theta_min = torch.tensor([-180,-120,-120,-180,-110,-360], dtype=torch.float32, device=device)*math.pi/180.0
theta_max = torch.tensor([ 180, 120, 120, 180, 110, 360], dtype=torch.float32, device=device)*math.pi/180.0

def dh(a, alpha, d, theta):
    """
    theta: [T], a/alpha/d: scalaires -> broadcastés à [T]
    Renvoie la matrice DH vectorisée [T,4,4]
    """
    theta = theta.view(-1)                           # [T]
    ct, st = torch.cos(theta), torch.sin(theta)      # [T], [T]
    a     = torch.as_tensor(a,     device=theta.device, dtype=theta.dtype)
    alpha = torch.as_tensor(alpha, device=theta.device, dtype=theta.dtype)
    d     = torch.as_tensor(d,     device=theta.device, dtype=theta.dtype)
    ca, sa = torch.cos(alpha), torch.sin(alpha)      # 0-D
    a_t  = torch.ones_like(theta) * a
    d_t  = torch.ones_like(theta) * d
    ca_t = torch.ones_like(theta) * ca
    sa_t = torch.ones_like(theta) * sa
    z_t  = torch.zeros_like(theta)
    o_t  = torch.ones_like(theta)
    T = torch.stack([
        torch.stack([ ct,        -st*ca_t,   st*sa_t,  a_t*ct], dim=-1),
        torch.stack([ st,         ct*ca_t,  -ct*sa_t,  a_t*st], dim=-1),
        torch.stack([ z_t,        sa_t,      ca_t,     d_t   ], dim=-1),
        torch.stack([ z_t,        z_t,       z_t,      o_t   ], dim=-1),
    ], dim=-2)
    return T

def fk_thetas(theta_seq):
    """
    theta_seq: [T,6] -> [T,3] positions effecteur
    """
    T_all = []
    for j in range(n_joints):
        Tj = dh(link_a[j], link_alpha[j], link_d[j], theta_seq[:, j])  # [T,4,4]
        T_all.append(Tj)
    Tcum = T_all[0]
    for j in range(1, n_joints):
        Tcum = torch.einsum("tij,tjk->tik", Tcum, T_all[j])
    return Tcum[:, :3, 3]

# ----------------------------- Encodeur laplacien d'angles (joint-space) avec NOMBRES COMPLEXES
class LaplaceJointEncoder(nn.Module):
    def __init__(self, n_joints=6, K=64, duration=2.0, max_s=0.05):
        super().__init__()
        self.n_joints = n_joints
        self.K = K
        self.duration = duration
        w = 2*math.pi*torch.arange(1, K+1, dtype=torch.float32)/duration
        self.register_buffer("omega", w)  # [K]
        
        # *** NOUVEAUTÉ : Amplitudes complexes c = a + ib ***
        # On stocke séparément real et imag pour l'optimisation PyTorch
        self.c_real = nn.Parameter(torch.zeros(n_joints, K))
        self.c_imag = nn.Parameter(torch.zeros(n_joints, K))
        
        self.theta0 = nn.Parameter(torch.zeros(n_joints))
        self.log_s = nn.Parameter(torch.full((K,), -5.0))  # softplus≈0.0067
        self.max_s = max_s

    def forward(self, t_grid):
        """
        Calcul avec nombres complexes:
        θ_j(t) = θ₀ + Σ_k Re[ c_jk · exp(-(s_k + i·ω_k)·t) ]
               = θ₀ + Σ_k exp(-s_k·t) · Re[ c_jk · exp(i·ω_k·t) ]
        
        où c_jk = a_jk + i·b_jk est l'amplitude complexe
        """
        T = t_grid.shape[0]
        t = t_grid.view(T, 1, 1)                      # [T,1,1]
        w = self.omega.view(1, 1, self.K)             # [1,1,K]
        s = torch.clamp(F.softplus(self.log_s), max=self.max_s).view(1,1,self.K)
        
        # Créer les amplitudes complexes à partir de real et imag
        c_real = self.c_real.view(1, self.n_joints, self.K)  # [1,J,K]
        c_imag = self.c_imag.view(1, self.n_joints, self.K)  # [1,J,K]
        c_complex = torch.complex(c_real, c_imag)      # [1,J,K] dtype=complex64
        
        # Terme de décroissance (réel)
        decay = torch.exp(-s * t)                      # [T,1,K] real
        
        # Exponentielle complexe : exp(i·ω·t) = cos(ω·t) + i·sin(ω·t)
        phase = w * t                                  # [T,1,K] real
        exp_iwt = torch.complex(torch.cos(phase), torch.sin(phase))  # [T,1,K] complex
        
        # Multiplication complexe avec broadcasting: [1,J,K] * [T,1,K] -> [T,J,K]
        complex_osc = c_complex * exp_iwt              # [T,J,K] complex
        
        # Appliquer la décroissance
        damped_osc = decay * complex_osc               # [T,J,K] complex
        
        # Somme sur les harmoniques et extraire la partie réelle
        series = damped_osc.sum(dim=-1).real           # [T,J] real
        
        theta = series + self.theta0                   # [T,J]
        return theta

    def damping(self):
        return torch.clamp(F.softplus(self.log_s), max=self.max_s)
    
    def complex_amplitudes(self):
        """Retourne les amplitudes complexes c_jk pour analyse"""
        return torch.complex(self.c_real, self.c_imag)
    
    def amplitude_magnitudes(self):
        """Retourne les magnitudes |c_jk|"""
        return torch.sqrt(self.c_real**2 + self.c_imag**2)
    
    def amplitude_phases(self):
        """Retourne les phases arg(c_jk) en radians"""
        return torch.atan2(self.c_imag, self.c_real)

# ----------------------------- Modèle & init
encoder = LaplaceJointEncoder(n_joints=6, K=args.n_units,
                              duration=args.duration, max_s=0.05).to(device)
with torch.no_grad():
    # Initialiser les amplitudes complexes à zéro
    encoder.c_real.mul_(0.0)
    encoder.c_imag.mul_(0.0)
    encoder.theta0[:] = torch.tensor([0, 20, 40, -30, 10, 0],
                                     dtype=torch.float32)*math.pi/180.0

t = torch.tensor(t_np, dtype=torch.float32, device=device)

# Debug: vérifier la position initiale
with torch.no_grad():
    theta_init = encoder(t)
    pos_init = fk_thetas(theta_init)
    print(f"Position initiale (premier point): {pos_init[0].cpu().numpy()}")
    print(f"Position cible   (premier point): {target[0].cpu().numpy()}")
    print(f"Position initiale moyenne Z: {pos_init[:, 2].mean().item():.2f}")
    print(f"Position cible    moyenne Z: {target[:, 2].mean().item():.2f}")

# ----------------------------- Entraînement
opt = torch.optim.Adam(encoder.parameters(), lr=args.lr)

def loss_fn(theta_pred, target_xyz):
    pos = fk_thetas(theta_pred)  # [T,3]
    
    # Loss principale: traiter X, Y, Z de la même manière
    recon = F.mse_loss(pos, target_xyz)
    
    # Pour trajectoires plates, ajouter une contrainte forte sur Z constant
    if args.trajectory in ["circle_flat", "lemniscate"]:
        # Calculer la variance de Z (devrait être ~0 pour trajectoire plate)
        z_target_var = torch.var(target_xyz[:, 2])  # ~0 pour trajectoire plate
        z_pred_var = torch.var(pos[:, 2])
        # Forcer la variance prédite à être aussi petite que la variance cible
        z_consistency = torch.abs(z_pred_var - z_target_var)
        recon = recon + 10.0 * z_consistency
    
    dtheta = theta_pred[1:] - theta_pred[:-1]
    smooth = (dtheta**2).mean()
    over_min = torch.clamp(theta_min - theta_pred, min=0.0)
    over_max = torch.clamp(theta_pred - theta_max, min=0.0)
    limits = (over_min**2 + over_max**2).mean()
    s = encoder.damping()
    reg_s = s.mean()
    return recon + 0.01*smooth + 0.001*limits + 0.01*reg_s, pos

print("🚀 Entraînement (joint-space, FK only, COMPLEX)…")
for ep in range(1, args.epochs+1):
    opt.zero_grad()
    theta = encoder(t)                 # [T,6]
    L, pos = loss_fn(theta, target)
    L.backward(); opt.step()
    if ep % 100 == 0 or ep == 1:
        with torch.no_grad():
            err = torch.sqrt(((pos - target)**2).sum(dim=1)).mean().item()
            z_err = torch.abs(pos[:, 2] - target[:, 2]).mean().item()
            z_std = torch.std(pos[:, 2]).item()
            # Statistiques sur les amplitudes complexes
            magnitudes = encoder.amplitude_magnitudes()
            avg_mag = magnitudes.mean().item()
            max_mag = magnitudes.max().item()
        print(f"Epoch {ep:4d} | Loss {L.item():.5f} | mean |Δp| = {err:.3f} | Z_err = {z_err:.4f} | Z_std = {z_std:.4f} | |c|_avg = {avg_mag:.4f} | |c|_max = {max_mag:.4f}")

# Sanity checks
theta = encoder(t).detach()
assert torch.isfinite(theta).all(), "NaN/Inf dans theta"
pos = fk_thetas(theta).detach().cpu().numpy()
assert np.isfinite(pos).all(), "NaN/Inf dans FK"
target_v = target.detach().cpu().numpy()

# ----------------------------- Visualisation (trajet)
fig = plt.figure(figsize=(16,10))

# Subplot 1: Trajectoire 3D
ax = fig.add_subplot(231, projection='3d')
ax.set_title("Joint-Space Laplace FK — Trajectoire (COMPLEX)")
ax.plot(target_v[:,0], target_v[:,1], target_v[:,2], c='lightgray', lw=2, label="Target")
ax.plot(pos[:,0], pos[:,1], pos[:,2], c='red', lw=2, label="Drawing")

# Aspect 1:1:1
mins = np.min(np.vstack([target_v, pos]), axis=0)
maxs = np.max(np.vstack([target_v, pos]), axis=0)
ranges = maxs - mins
mid = (maxs + mins)/2
radius = ranges.max()/2 * 1.15
ax.set_xlim(mid[0]-radius, mid[0]+radius)
ax.set_ylim(mid[1]-radius, mid[1]+radius)
ax.set_zlim(mid[2]-radius, mid[2]+radius)
ax.set_box_aspect((1,1,1))
ax.legend()

# Projections
ax_xy = fig.add_subplot(234); ax_xz = fig.add_subplot(235); ax_yz = fig.add_subplot(236)
ax_xy.plot(target_v[:,0], target_v[:,1], c='lightgray'); ax_xy.plot(pos[:,0], pos[:,1], c='red'); ax_xy.set_title("X-Y")
ax_xz.plot(target_v[:,0], target_v[:,2], c='lightgray'); ax_xz.plot(pos[:,0], pos[:,2], c='red'); ax_xz.set_title("X-Z")
ax_yz.plot(target_v[:,1], target_v[:,2], c='lightgray'); ax_yz.plot(pos[:,1], pos[:,2], c='red'); ax_yz.set_title("Y-Z")

# *** NOUVEAUTÉ : Visualisation des amplitudes complexes ***
ax_complex = fig.add_subplot(232)
ax_complex.set_title("Amplitudes complexes c_jk (plan ℂ)")
ax_complex.set_xlabel("Re(c)")
ax_complex.set_ylabel("Im(c)")
ax_complex.grid(True, alpha=0.3)
ax_complex.axhline(y=0, color='k', linewidth=0.5)
ax_complex.axvline(x=0, color='k', linewidth=0.5)

# Récupérer les amplitudes complexes
with torch.no_grad():
    c_complex = encoder.complex_amplitudes().cpu().numpy()  # [n_joints, K]
    
# Tracer chaque joint avec une couleur différente
colors = plt.cm.tab10(np.linspace(0, 1, n_joints))
for j in range(n_joints):
    c_j = c_complex[j]  # [K]
    ax_complex.scatter(c_j.real, c_j.imag, alpha=0.6, s=20, 
                      c=[colors[j]], label=f"Joint {j+1}")
ax_complex.legend(loc='upper right', fontsize=8)
ax_complex.set_aspect('equal', adjustable='box')

# *** NOUVEAUTÉ : Visualisation des magnitudes ***
ax_mag = fig.add_subplot(233)
ax_mag.set_title("Magnitudes |c_jk| par harmonique")
ax_mag.set_xlabel("Harmonique k")
ax_mag.set_ylabel("|c_jk|")
with torch.no_grad():
    mags = encoder.amplitude_magnitudes().cpu().numpy()  # [n_joints, K]
for j in range(n_joints):
    ax_mag.plot(mags[j], alpha=0.7, c=colors[j], label=f"Joint {j+1}")
ax_mag.legend(fontsize=8)
ax_mag.grid(True, alpha=0.3)

plt.tight_layout()

# ----------------------------- Animation bras (FK pur)
def fk_points(theta_frame):
    Tcur = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0)  # [1,4,4]
    pts = [torch.tensor([0,0,0], dtype=torch.float32, device=device)]
    for j in range(n_joints):
        Tj = dh(link_a[j], link_alpha[j], link_d[j], theta_frame[j].unsqueeze(0))
        Tcur = torch.einsum("bij,bjk->bik", Tcur, Tj)
        pts.append(Tcur[0,:3,3])
    return torch.stack(pts, dim=0).detach().cpu().numpy()  # [7,3]

fig2 = plt.figure(figsize=(8,8))
ax2 = fig2.add_subplot(111, projection='3d')
ax2.set_title("Bras — Animation (FK)")
ax2.set_box_aspect((1,1,1))
ax2.set_xlim(mid[0]-radius, mid[0]+radius)
ax2.set_ylim(mid[1]-radius, mid[1]+radius)
# S'assurer que z=0 (la base) est visible
z_min = min(mid[2]-radius, -5)
z_max = max(mid[2]+radius, mid[2]+radius)
ax2.set_zlim(z_min, z_max)

arm_lines = [ax2.plot([], [], [], 'b-', lw=5)[0] for _ in range(n_joints)]
joints = [ax2.plot([], [], [], 'ro', ms=6)[0] for _ in range(n_joints+1)]
trail, = ax2.plot([], [], [], '-', lw=3, c='red', alpha=0.85)

# Ajouter un plan au niveau z=0 pour visualiser la base
xx, yy = np.meshgrid(np.linspace(mid[0]-radius, mid[0]+radius, 10),
                      np.linspace(mid[1]-radius, mid[1]+radius, 10))
zz = np.zeros_like(xx)
ax2.plot_surface(xx, yy, zz, alpha=0.1, color='gray')

drawn = []

def init_anim():
    th0 = theta[0]
    pts0 = fk_points(th0)  # [7,3]
    for j in range(n_joints):
        p0, p1 = pts0[j], pts0[j+1]
        arm_lines[j].set_data([p0[0], p1[0]], [p0[1], p1[1]])
        arm_lines[j].set_3d_properties([p0[2], p1[2]])
    for j in range(n_joints+1):
        p = pts0[j]
        joints[j].set_data([p[0]], [p[1]])
        joints[j].set_3d_properties([p[2]])
    trail.set_data([pts0[-1][0]], [pts0[-1][1]])
    trail.set_3d_properties([pts0[-1][2]])
    drawn.clear()
    drawn.append(pts0[-1])
    # vue statique de départ
    ax2.view_init(elev=25, azim=45)
    return arm_lines + joints + [trail]

def animate(i):
    th = theta[i % T]
    pts = fk_points(th)
    for j in range(n_joints):
        p0, p1 = pts[j], pts[j+1]
        arm_lines[j].set_data([p0[0], p1[0]], [p0[1], p1[1]])
        arm_lines[j].set_3d_properties([p0[2], p1[2]])
    for j in range(n_joints+1):
        p = pts[j]
        joints[j].set_data([p[0]],[p[1]])
        joints[j].set_3d_properties([p[2]])
    ee = pts[-1]
    drawn.append(ee)
    if len(drawn) > 1200: drawn.pop(0)
    arr = np.array(drawn)
    trail.set_data(arr[:,0], arr[:,1]); trail.set_3d_properties(arr[:,2])

    # rotation de caméra optionnelle
    if args.spin:
        ax2.view_init(elev=25, azim=(i*0.7)%360)

    return arm_lines + joints + [trail]

# Variable pour stocker l'animation courante
current_anim = [None]

def start_animation(event):
    """Relance l'animation depuis le début"""
    # Arrêter l'animation en cours si elle existe
    try:
        if current_anim[0] is not None:
            current_anim[0].event_source.stop()
    except (AttributeError, TypeError):
        pass  # Pas d'animation à arrêter
    
    # Réinitialiser la trajectoire dessinée
    drawn.clear()
    
    # Créer une nouvelle animation
    new_anim = FuncAnimation(fig2, animate, frames=T, init_func=init_anim,
                             interval=20, blit=False, repeat=False)
    current_anim[0] = new_anim
    ANIMS.append(new_anim)
    fig2.canvas.draw_idle()

# Ajouter le bouton Start
from matplotlib.widgets import Button
ax_button = plt.axes([0.81, 0.02, 0.1, 0.05])  # [left, bottom, width, height]
btn_start = Button(ax_button, 'Start')
btn_start.on_clicked(start_animation)

print("🎬 Animation… Cliquez sur 'Start' pour relancer. Ferme la figure pour terminer.")
# Lancer la première animation
start_animation(None)
plt.show()