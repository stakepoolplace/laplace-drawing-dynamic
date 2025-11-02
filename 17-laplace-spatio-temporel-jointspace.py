#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Joint-Space Laplace Driver (Spatio-Temporal version)
Author : Eric Marchand
Date   : 2025-11-01

- 6 revolute joints
- θ_j(t) = Σ_{k,m} Re[ A_km · e^{-(s_k+iω_k)t} · φ_m(j) ] + θ0
- Spectral coupling between joints via spatial Laplace basis φ_m(j)
- Training: MSE on end-effector position (FK only)
- Visualization: trajectory, spectra, and animated arm with Start / Pause controls
"""

import matplotlib
try:
    matplotlib.use("MacOSX")
except Exception:
    pass

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import argparse, math, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F

ANIMS = []

# ----------------------------- Args
p = argparse.ArgumentParser()
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
p.add_argument("--epochs", type=int, default=3000)
p.add_argument("--lr", type=float, default=1e-2)
p.add_argument("--n_units", type=int, default=64)
p.add_argument("--duration", type=float, default=2.0)
p.add_argument("--n_points", type=int, default=400)
p.add_argument("--trajectory", type=str, default="circle_flat",
               choices=["circle_flat","circle_wavy","lemniscate"])
p.add_argument("--spin", action="store_true")
p.add_argument("--seed", type=int, default=123)
args = p.parse_args()

torch.manual_seed(args.seed); np.random.seed(args.seed)
device = torch.device(args.device)
print(f"Device: {device}")

# ----------------------------- Target Trajectories
def generate_trajectory(kind="circle_flat", n=400, duration=2.0):
    t = np.linspace(0, duration, n)
    if kind == "circle_flat":
        R = 60.0
        x = R*np.cos(2*np.pi*t/duration)
        y = R*np.sin(2*np.pi*t/duration)
        z = np.full_like(x, 50.0)
    elif kind == "circle_wavy":
        R = 60.0
        x = R*np.cos(2*np.pi*t/duration)
        y = R*np.sin(2*np.pi*t/duration)
        z = 50.0 + 10.0*np.sin(4*np.pi*t/duration)
    else:
        a = 60.0
        s = np.linspace(-np.pi/2, 3*np.pi/2, n)
        x = a*np.sin(s)/(1+np.cos(s)**2)
        y = a*np.sin(s)*np.cos(s)/(1+np.cos(s)**2)
        z = np.full_like(x, 50.0)
    return np.stack([x,y,z], axis=-1), t

target_np, t_np = generate_trajectory(args.trajectory, args.n_points, args.duration)
target = torch.tensor(target_np, dtype=torch.float32, device=device)
T = target.shape[0]

print(f"📊 Target trajectory '{args.trajectory}' with {T} points.")

# ----------------------------- Arm (6-DoF, DH parameters)
link_a    = [0.0,  30.0, 25.0,  0.0,  0.0,  0.0]
link_alpha= [np.pi/2, 0.0, 0.0, np.pi/2, -np.pi/2, 0.0]
link_d    = [0.0,   0.0,  0.0, 35.0,    0.0,   15.0]
n_joints = 6

def dh(a, alpha, d, theta):
    ct, st = torch.cos(theta), torch.sin(theta)
    ca = torch.cos(torch.tensor(alpha, device=theta.device, dtype=theta.dtype))
    sa = torch.sin(torch.tensor(alpha, device=theta.device, dtype=theta.dtype))
    a_t = torch.full_like(ct, a)
    d_t = torch.full_like(ct, d)
    return torch.stack([
        torch.stack([ct, -st*ca,  st*sa, a_t*ct], dim=-1),
        torch.stack([st,  ct*ca, -ct*sa, a_t*st], dim=-1),
        torch.stack([torch.zeros_like(ct), sa.expand_as(ct), ca.expand_as(ct), d_t], dim=-1),
        torch.stack([torch.zeros_like(ct), torch.zeros_like(ct), torch.zeros_like(ct), torch.ones_like(ct)], dim=-1),
    ], dim=-2)

def fk_thetas(theta_seq):
    T_all = []
    for j in range(n_joints):
        Tj = dh(link_a[j], link_alpha[j], link_d[j], theta_seq[:, j])
        T_all.append(Tj)
    Tcum = T_all[0]
    for j in range(1, n_joints):
        Tcum = torch.einsum("tij,tjk->tik", Tcum, T_all[j])
    return Tcum[:, :3, 3]

# ----------------------------- SPATIO-TEMPORAL ENCODER
class LaplaceJointSpatioTemporal(nn.Module):
    def __init__(self, n_joints=6, Kt=32, Kx=8, duration=2.0, max_s=0.05):
        super().__init__()
        self.n_joints = n_joints
        self.Kt = Kt
        self.Kx = Kx
        self.duration = duration

        w = 2*math.pi*torch.arange(1, Kt+1, dtype=torch.float32)/duration
        self.register_buffer("omega", w)
        self.log_s = nn.Parameter(torch.full((Kt,), -4.0))
        self.max_s = max_s

        j_idx = torch.arange(1, n_joints+1, dtype=torch.float32)
        phi = []
        for m in range(1, Kx//2 + 1):
            phi.append(torch.sin(m*j_idx))
            phi.append(torch.cos(m*j_idx))
        phi = torch.stack(phi, dim=1)
        phi = phi / torch.norm(phi, dim=0, keepdim=True)
        self.register_buffer("phi_joint", phi)

        self.A_real = nn.Parameter(torch.randn(Kt, Kx) * 0.01)
        self.A_imag = nn.Parameter(torch.randn(Kt, Kx) * 0.01)
        self.theta0 = nn.Parameter(torch.zeros(n_joints))

    def forward(self, t_grid):
        T = t_grid.shape[0]
        t = t_grid.view(T, 1, 1)
        s = torch.clamp(F.softplus(self.log_s), max=self.max_s)
        w = self.omega.view(1, 1, self.Kt)
        phi = self.phi_joint.to(torch.cfloat)
        exp_t = torch.exp(-t * (s.view(1,1,self.Kt) + 1j*w))
        A = torch.complex(self.A_real, self.A_imag)
        spec = torch.einsum('tk,kx,xj->tj', exp_t.squeeze(1), A, phi.T)
        return spec.real + self.theta0

    def damping(self):
        return torch.clamp(F.softplus(self.log_s), max=self.max_s)

# ----------------------------- Model + Training
encoder = LaplaceJointSpatioTemporal(n_joints=6, Kt=args.n_units, Kx=8,
                                     duration=args.duration).to(device)
t = torch.tensor(t_np, dtype=torch.float32, device=device)
opt = torch.optim.Adam(encoder.parameters(), lr=args.lr)

def loss_fn(theta_pred, target_xyz):
    pos = fk_thetas(theta_pred)
    recon = F.mse_loss(pos, target_xyz)
    dtheta = theta_pred[1:] - theta_pred[:-1]
    smooth = (dtheta**2).mean()
    reg_s = encoder.damping().mean()
    return recon + 0.01*smooth + 0.01*reg_s, pos

print("🚀 Training Laplace Spatio-Temporal Encoder…")
for ep in range(1, args.epochs+1):
    opt.zero_grad()
    theta = encoder(t)
    L, pos = loss_fn(theta, target)
    L.backward()
    opt.step()
    if ep % 100 == 0:
        err = torch.sqrt(((pos - target)**2).sum(dim=1)).mean().item()
        print(f"Epoch {ep:4d} | Loss {L.item():.5f} | mean |Δp| = {err:.3f}")

# ----------------------------- Visualization
theta = encoder(t).detach()
pos = fk_thetas(theta).detach().cpu().numpy()
target_v = target.detach().cpu().numpy()

fig = plt.figure(figsize=(14,8))
ax = fig.add_subplot(121, projection='3d')
ax.plot(target_v[:,0], target_v[:,1], target_v[:,2], c='gray', lw=2, label='Target')
ax.plot(pos[:,0], pos[:,1], pos[:,2], c='red', lw=2, label='Prediction')
ax.legend()
ax.set_title("Trajectory Comparison")

ax2 = fig.add_subplot(122)
A_mag = torch.sqrt(encoder.A_real.detach()**2 + encoder.A_imag.detach()**2).cpu().numpy()
ax2.imshow(A_mag, aspect='auto', cmap='magma')
ax2.set_title("|A_km| (Temporal × Spatial Spectrum)")
ax2.set_xlabel("Spatial mode m")
ax2.set_ylabel("Temporal mode k")
plt.tight_layout()

# ----------------------------- Animation (FK Arm)
def dh_frame(a, alpha, d, theta):
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([[ct, -st*ca,  st*sa, a*ct],
                     [st,  ct*ca, -ct*sa, a*st],
                     [0, sa, ca, d],
                     [0, 0, 0, 1]])

def fk_points_numpy(theta_frame):
    Tcur = np.eye(4)
    pts = [np.zeros(3)]
    for j in range(n_joints):
        Tcur = Tcur @ dh_frame(link_a[j], link_alpha[j], link_d[j], theta_frame[j])
        pts.append(Tcur[:3,3])
    return np.stack(pts, axis=0)

fig2 = plt.figure(figsize=(7,7))
ax3 = fig2.add_subplot(111, projection='3d')
ax3.set_title("Arm Animation (Spatio-Temporal Laplace)")
ax3.set_box_aspect((1,1,1))
trail, = ax3.plot([], [], [], '-', c='red', lw=3, alpha=0.8)
lines = [ax3.plot([], [], [], 'b-', lw=5)[0] for _ in range(n_joints)]
points = [ax3.plot([], [], [], 'ro', ms=6)[0] for _ in range(n_joints+1)]
drawn = []

mins, maxs = np.min(pos,0), np.max(pos,0)
mid, radius = (maxs+mins)/2, (maxs-mins).max()/2*1.2
ax3.set_xlim(mid[0]-radius, mid[0]+radius)
ax3.set_ylim(mid[1]-radius, mid[1]+radius)
ax3.set_zlim(mid[2]-radius, mid[2]+radius)

def init_anim():
    drawn.clear()
    return lines + points + [trail]

def animate(i):
    th = theta[i%T].cpu().numpy()
    pts = fk_points_numpy(th)
    for j in range(n_joints):
        p0,p1 = pts[j], pts[j+1]
        lines[j].set_data([p0[0],p1[0]],[p0[1],p1[1]])
        lines[j].set_3d_properties([p0[2],p1[2]])
    for j in range(n_joints+1):
        p=pts[j]; points[j].set_data([p[0]],[p[1]]); points[j].set_3d_properties([p[2]])
    drawn.append(pts[-1])
    arr=np.array(drawn[-300:])
    trail.set_data(arr[:,0], arr[:,1]); trail.set_3d_properties(arr[:,2])
    if args.spin: ax3.view_init(25,(i*0.7)%360)
    return lines + points + [trail]

# ----------------------------- Buttons (Start / Pause)
ANIM = [None]
paused = [False]

def start_anim(event=None):
    """(Re)start the animation from the beginning."""
    if ANIM[0] is not None and hasattr(ANIM[0], "event_source"):
        try:
            ANIM[0].event_source.stop()
        except Exception:
            pass
    new_anim = FuncAnimation(fig2, animate, frames=T, init_func=init_anim,
                             interval=20, blit=False, repeat=False)
    ANIM[0] = new_anim
    ANIMS.append(new_anim)
    fig2.canvas.draw_idle()

def pause_anim(event=None):
    """Toggle pause/resume state."""
    if ANIM[0] is None or not hasattr(ANIM[0], "event_source"):
        return
    paused[0] = not paused[0]
    if paused[0]:
        ANIM[0].event_source.stop()
        btn_pause.label.set_text("Resume")
    else:
        ANIM[0].event_source.start()
        btn_pause.label.set_text("Pause")

# Buttons
ax_btn_start = plt.axes([0.75, 0.02, 0.1, 0.05])
ax_btn_pause = plt.axes([0.87, 0.02, 0.1, 0.05])
btn_start = Button(ax_btn_start, "Start")
btn_pause = Button(ax_btn_pause, "Pause")
btn_start.on_clicked(start_anim)
btn_pause.on_clicked(pause_anim)

print("🎬 Click 'Start' to animate, 'Pause' to toggle.")
start_anim()
plt.show()
