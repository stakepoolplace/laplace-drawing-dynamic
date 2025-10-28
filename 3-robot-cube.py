# =============================================================================
# 9-3D-CUBE-ROBOT-ARM-CONTINUOUS.py
# Le bras trace tout le temps, sans lever le stylo
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn

# ===============================
# 1. Cube complet (toutes arêtes)
# ===============================
def generate_cube_points(size=80, n_per_edge=50, z_offset=80):
    s = size / 2
    pts = []
    edges = [
        # base z = -s
        [[-s,-s,-s],[ s,-s,-s]],
        [[ s,-s,-s],[ s, s,-s]],
        [[ s, s,-s],[-s, s,-s]],
        [[-s, s,-s],[-s,-s,-s]],
        # haut z = s
        [[-s,-s, s],[ s,-s, s]],
        [[ s,-s, s],[ s, s, s]],
        [[ s, s, s],[-s, s, s]],
        [[-s, s, s],[-s,-s, s]],
        # verticales
        [[-s,-s,-s],[-s,-s, s]],
        [[ s,-s,-s],[ s,-s, s]],
        [[ s, s,-s],[ s, s, s]],
        [[-s, s,-s],[-s, s, s]]
    ]
    for e0, e1 in edges:
        line = np.linspace(e0, e1, n_per_edge)
        pts.append(line)
    pts = np.concatenate(pts)
    pts[:, 2] += z_offset
    return pts, edges, s, z_offset

cube_points, cube_edges, cube_half, cube_z = generate_cube_points(z_offset=80)
N = len(cube_points)

# ===============================
# 2. Modèle LSTM (placeholder)
# ===============================
x, y, z = cube_points.T
features = np.stack([x, y, z], axis=1)
target = np.ones((N, 1))
X_seq = torch.tensor(features, dtype=torch.float32)
y_seq = torch.tensor(target, dtype=torch.float32)

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

# ===============================
# 3. Bras 6-DoF
# ===============================
link_lengths = [40, 35, 30, 25, 20, 15]
n_joints = len(link_lengths)
total_height = sum(link_lengths)

def ik_6dof_with_z(tx, ty, tz):
    cx, cy, cz = 0.0, 0.0, total_height
    positions = [np.array([cx, cy, cz])]
    for L in link_lengths:
        dx, dy, dz = tx - cx, ty - cy, tz - cz
        dist = np.linalg.norm([dx, dy, dz])
        if dist < 1e-6: dist = 1e-6
        dir = np.array([dx, dy, dz]) / dist
        step = dir * min(L, dist)
        cx += step[0]; cy += step[1]; cz += step[2]
        positions.append(np.array([cx, cy, cz]))
    return positions

# ===============================
# 4. Setup 3D
# ===============================
fig = plt.figure(figsize=(13,10))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-100,100); ax.set_ylim(-100,100); ax.set_zlim(0,160)
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.view_init(elev=25, azim=-60)

# --- Cube fil de fer fixe ---
s = cube_half
for e0, e1 in cube_edges:
    x = [e0[0], e1[0]]
    y = [e0[1], e1[1]]
    z = [e0[2] + cube_z, e1[2] + cube_z]
    ax.plot(x, y, z, color='gray', linestyle='--', linewidth=1.2, alpha=0.6)

# --- Bras ---
arm_lines = [ax.plot([], [], [], 'b-', linewidth=6)[0] for _ in range(n_joints)]
joint_lines = [ax.plot([], [], [], 'ro', markersize=10)[0] for _ in range(n_joints+1)]
drawing_line, = ax.plot([], [], [], 'k-', linewidth=3)

trail = []
rnn_hidden = None

# ===============================
# 5. Animation
# ===============================
def animate(frame):
    global rnn_hidden
    idx = frame % N
    tx, ty, tz = cube_points[idx]
    positions = ik_6dof_with_z(tx, ty, tz)
    pos_array = np.array(positions)

    # --- Segments du bras ---
    for i, line in enumerate(arm_lines):
        p0, p1 = pos_array[i], pos_array[i+1]
        line.set_data([p0[0], p1[0]], [p0[1], p1[1]])
        line.set_3d_properties([p0[2], p1[2]])

    # --- Boules rouges ---
    for i, line in enumerate(joint_lines):
        x, y, z = pos_array[i]
        line.set_data([x], [y])
        line.set_3d_properties([z])

    # --- Tracé continu (toujours) ---
    trail.append([tx, ty, tz])
    tnp = np.array(trail)
    drawing_line.set_data(tnp[:,0], tnp[:,1])
    drawing_line.set_3d_properties(tnp[:,2])

    return arm_lines + joint_lines + [drawing_line]

# ===============================
# 6. Lancement
# ===============================
print("Starting 3D CUBE animation (tracé continu)...")
anim = FuncAnimation(fig, animate, frames=2000, interval=15, blit=False)
plt.tight_layout()
plt.show()

