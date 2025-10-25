# =============================================================================
# 9-3D-fourier-robot-arm-RED-BALLS-FINAL.py
# CHAQUE ROTULE = BOULE ROUGE (VISIBLE !)
# FIX: .set_data_3d() au lieu de _offsets3d
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from skimage import io, color, measure
from scipy import ndimage
import torch
import torch.nn as nn

# ===============================
# 1. Chargement + contour
# ===============================
print("Loading face.png...")
img = io.imread("face.png")
if img.shape[-1] == 4: img = img[..., :3]
gray = color.rgb2gray(img)
edges = gray < 0.5
edges = ndimage.binary_fill_holes(edges)
contours = measure.find_contours(edges, 0.8)
points = np.concatenate(contours)
x, y = points[:, 1], -points[:, 0]
x -= x.mean(); y -= y.mean()
z_contour = x + 1j * y

# Densification
N = 6000
z_dense = np.interp(np.linspace(0, len(z_contour), N), np.arange(len(z_contour)), z_contour)

# Mise à l'échelle
R = np.max(np.abs(z_dense))
scale = 120.0 / R
z = z_dense * scale
print(f"Scaled contour: radius = {np.max(np.abs(z)):.1f}")

# ===============================
# 2. Courbure + LSTM
# ===============================
dx = np.gradient(z.real); dy = np.gradient(z.imag)
ddx = np.gradient(dx); ddy = np.gradient(dy)
curvature = np.abs(dx*ddy - dy*ddx) / (dx**2 + dy**2 + 1e-8)**1.5
curvature /= (curvature.max() + 1e-8)

speed_magnitude = np.abs(dx + 1j*dy).flatten()
features = np.stack([curvature, np.gradient(curvature), speed_magnitude], axis=1)
target = 1 / (1 + 3*curvature)[:, None]

X_seq = torch.tensor(features, dtype=torch.float32)
y_seq = torch.tensor(target, dtype=torch.float32)

# LSTM
class SpeedLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(3, 16, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(16, 1), nn.Sigmoid())
    def forward(self, x, h=None):
        if x.dim() == 2: x = x.unsqueeze(0)
        out, h = self.lstm(x, h)
        return self.fc(out), h

model = SpeedLSTM()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
Xb = X_seq.unsqueeze(0); yb = y_seq.unsqueeze(0)
print("Training LSTM...")
for epoch in range(100):
    opt.zero_grad()
    out, _ = model(Xb)
    loss = loss_fn(out, yb)
    loss.backward()
    opt.step()
print(f"Training done. Loss: {loss.item():.6f}")

# ===============================
# 3. FFT complète
# ===============================
c = np.fft.fft(z) / N
freqs = np.fft.fftfreq(N, 1/N)

# ===============================
# 4. Bras 6-DoF
# ===============================
link_lengths = [40, 35, 30, 25, 20, 15]
n_joints = len(link_lengths)
total_height = sum(link_lengths)

# ===============================
# 5. IK avec Z décroissant
# ===============================
def ik_6dof_with_z(target_x, target_y):
    cx, cy, cz = 0.0, 0.0, total_height
    positions = [np.array([cx, cy, cz])]

    for L in link_lengths:
        dx = target_x - cx
        dy = target_y - cy
        dist_xy = np.hypot(dx, dy)
        angle_xy = np.arctan2(dy, dx) if dist_xy > 1e-6 else 0.0
        step_xy = min(L, dist_xy)
        cx += step_xy * np.cos(angle_xy)
        cy += step_xy * np.sin(angle_xy)
        cz -= L
        positions.append(np.array([cx, cy, cz]))
    return positions

# ===============================
# 6. Setup 3D
# ===============================
fig = plt.figure(figsize=(13, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-140, 140); ax.set_ylim(-140, 140); ax.set_zlim(0, total_height + 20)
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.view_init(elev=25, azim=-60)

# Segments du bras (bleus)
arm_lines = [ax.plot([], [], [], 'b-', linewidth=6, solid_capstyle='round')[0] for _ in range(n_joints)]

# ROTULES ROUGES : une ligne par boule (7 boules : base + 6 rotules)
joint_lines = []
sizes = [12] * (n_joints + 1)
sizes[-1] = 16  # stylo plus gros
for size in sizes:
    line, = ax.plot([], [], [], 'ro', markersize=size, alpha=0.9)
    joint_lines.append(line)

# Plan de dessin
xx, yy = np.meshgrid(np.linspace(-140,140,12), np.linspace(-140,140,12))
ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.4, color='lightgray', zorder=0)

# Dessin
drawing_line, = ax.plot([], [], [], 'k-', linewidth=4, zorder=1)

trail = []
rnn_hidden = None

# ===============================
# 7. Animation
# ===============================
def animate(frame):
    global rnn_hidden
    t = 2 * np.pi * (frame % N) / N
    Z = 0.0 + 0.0j

    # --- Vitesse LSTM ---
    idx = frame % N
    feat = X_seq[idx:idx+1].unsqueeze(0)
    model.eval()
    with torch.no_grad():
        speed_pred, rnn_hidden = model(feat, rnn_hidden)
    accel = 0.5 + 0.8 * speed_pred.item()

    # --- Reconstruction ---
    for k in range(-800, 801):
        i = k % N
        sigma = 0.001 * np.sin(0.2*t + 0.1*k)
        omega = 2*np.pi*freqs[i]*accel
        Z += c[i] * np.exp((sigma + 1j*omega)*t)

    trail.append(Z)

    # --- IK ---
    tx, ty = Z.real, Z.imag
    positions = ik_6dof_with_z(tx, ty)
    pos_array = np.array(positions)

    # --- Mise à jour segments ---
    for i, line in enumerate(arm_lines):
        p0, p1 = pos_array[i], pos_array[i+1]
        line.set_data([p0[0], p1[0]], [p0[1], p1[1]])
        line.set_3d_properties([p0[2], p1[2]])

    # --- Mise à jour BOULES ROUGES (une par une) ---
    for i, line in enumerate(joint_lines):
        x, y, z = pos_array[i]
        line.set_data([x], [y])
        line.set_3d_properties([z])

    # --- Dessin ---
    txs = [p.real for p in trail[-N:]]
    tys = [p.imag for p in trail[-N:]]
    tzs = [0.0] * len(txs)
    drawing_line.set_data(txs, tys)
    drawing_line.set_3d_properties(tzs)

    return arm_lines + joint_lines + [drawing_line]

# ===============================
# 8. Lancement
# ===============================
print("Starting RED BALLS 6-DoF animation... (infinite loop)")
anim = FuncAnimation(fig, animate, frames=None, interval=15, blit=False, repeat=True)
plt.tight_layout()
plt.show()