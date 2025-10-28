# =============================================================================
# robot-arm-fourier-lstm-ENHANCED.py
# Amélioration de l'apprentissage de la forme globale (face2.png)
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from skimage import io, color, measure, morphology
from scipy import ndimage, signal
import torch
import torch.nn as nn

# ===============================
# 0. PARAMÈTRES GÉNÉRAUX
# ===============================
win, hop = 256, 32        # Fenêtre plus large → contexte global
n_keep = 32               # Plus de coefficients conservés
N = 2048                  # Plus de points pour la forme

# ===============================
# 1. CHARGEMENT + CONTOUR
# ===============================
img = io.imread("face2.png")
if len(img.shape) == 3:
    if img.shape[-1] == 4: img = img[..., :3]
    gray = color.rgb2gray(img)
else:
    gray = img.astype(float) / 255.0

binary = gray < 0.5
binary = ndimage.binary_fill_holes(binary)
binary = morphology.remove_small_objects(binary, min_size=100)

labels, num_labels = ndimage.label(binary)
sizes = ndimage.sum(binary, labels, range(1, num_labels + 1))
main_label = np.argmax(sizes) + 1
binary = labels == main_label

contours = measure.find_contours(binary, 0.5)
contour = max(contours, key=len)
y, x = contour[:, 0], contour[:, 1]
x -= x.mean(); y -= y.mean()

angles = np.arctan2(y, x)
order = np.argsort(angles)
x, y = x[order], y[order]
x = np.append(x, x[0])
y = np.append(y, y[0])
z_contour = x + 1j * y

# Densification
dz = np.diff(z_contour)
distances = np.abs(dz)
cumdist = np.cumsum(distances)
cumdist = np.insert(cumdist, 0, 0.0)
t_old = cumdist / cumdist[-1]
t_new = np.linspace(0, 1, N)

z_dense = np.interp(t_new, t_old, np.real(z_contour)) + 1j * np.interp(t_new, t_old, np.imag(z_contour))
z_dense -= np.mean(z_dense)
R = np.max(np.abs(z_dense))
z = (z_dense / R) * 120.0

# ===============================
# 2. FONCTION DE FEATURES (STFT)
# ===============================
def stft_features(sig):
    window = np.hanning(win)
    feats = []
    for i in range(0, len(sig) - win + 1, hop):
        seg = sig[i:i+win] * window
        fft_seg = np.fft.fft(seg)
        amp = np.abs(fft_seg[:n_keep])
        phase = np.unwrap(np.angle(fft_seg[:n_keep]))
        feats.append(np.stack([amp, phase], axis=1))
    return np.array(feats)

v = np.gradient(z)
a = np.gradient(v)

feat_v = stft_features(v)
feat_a = stft_features(a)
feat_z = stft_features(z)

L = min(len(feat_v), len(feat_a), len(feat_z))
X_raw = np.concatenate([feat_v[:L], feat_a[:L]], axis=2)
y_raw = feat_z[:L]

# ===============================
# 3. FEATURES GLOBALES
# ===============================
radius = np.abs(z)
angle = np.unwrap(np.angle(z))
r_mean = np.mean(radius)
r_std = np.std(radius)
a_std = np.std(angle)
global_context = np.array([r_mean, r_std, a_std], dtype=np.float32)

context_tiled = np.tile(global_context, (L, X_raw.shape[1], 1))
X_raw = np.concatenate([X_raw, context_tiled], axis=2)

# Normalisation
X_mean = X_raw.mean(axis=(0,1), keepdims=True)
X_std = X_raw.std(axis=(0,1), keepdims=True) + 1e-8
y_mean = y_raw.mean(axis=(0,1), keepdims=True)
y_std = y_raw.std(axis=(0,1), keepdims=True) + 1e-8

X = torch.tensor((X_raw - X_mean) / X_std, dtype=torch.float32)
y = torch.tensor((y_raw - y_mean) / y_std, dtype=torch.float32)

# ===============================
# 4. MODÈLE LSTM AMÉLIORÉ
# ===============================
class BiFreqLSTM(nn.Module):
    def __init__(self, input_dim, hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, batch_first=True, bidirectional=True, num_layers=2, dropout=0.3)
        self.fc = nn.Linear(hidden * 2, 2)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)

model = BiFreqLSTM(X.shape[2])
opt = torch.optim.Adam(model.parameters(), lr=2e-3)
loss_fn = nn.MSELoss()

# ===============================
# 5. ENTRAÎNEMENT (AVEC DOUBLE PERTE)
# ===============================
print("Entraînement amélioré...")
best_loss = float('inf')
wait = 0
for epoch in range(6000):
    opt.zero_grad()
    out = model(X)
    loss_freq = loss_fn(out, y)

    # Reconstruction géométrique rapide
    pred_geo = torch.fft.ifft(torch.fft.fft(out, n=win)).real
    target_geo = torch.fft.ifft(torch.fft.fft(y, n=win)).real
    loss_geo = torch.mean((pred_geo - target_geo) ** 2)

    loss = loss_freq + 0.2 * loss_geo
    loss.backward()
    opt.step()

    if epoch % 300 == 0:
        print(f"Epoch {epoch:4d} | Loss = {loss.item():.6f}")

    if loss.item() < best_loss:
        best_loss = loss.item()
        wait = 0
    else:
        wait += 1
        if wait >= 300:
            print("Early stopping")
            break

# ===============================
# 6. RECONSTRUCTION
# ===============================
model.eval()
with torch.no_grad():
    pred = model(X).numpy() * y_std + y_mean

reconstructed = np.zeros(len(z), dtype=complex)
count = np.zeros(len(z))
window = np.hanning(win)

for i in range(len(pred)):
    start = i * hop
    amp, phase = pred[i, :, 0], pred[i, :, 1]
    spec = np.zeros(win, dtype=complex)
    spec[:n_keep] = amp * np.exp(1j * phase)
    if n_keep > 1:
        spec[-n_keep+1:] = np.conj(spec[1:n_keep][::-1])
    seg = np.fft.ifft(spec)
    seg *= window
    end = min(start + win, len(z))
    seg = seg[:end - start]
    reconstructed[start:end] += seg.astype(complex)
    count[start:end] += window[:len(seg)]

count[count == 0] = 1
z_rec = reconstructed / count

# Lissage doux
z_rec_real = signal.savgol_filter(z_rec.real, 51, 3)
z_rec_imag = signal.savgol_filter(z_rec.imag, 51, 3)
z_rec = z_rec_real + 1j * z_rec_imag
z_rec -= np.mean(z_rec)
R_rec = np.max(np.abs(z_rec))
z_rec = (z_rec / R_rec) * 120.0 if R_rec > 1e-6 else z_rec

# ===============================
# 7. VISUALISATION COMPARATIVE
# ===============================
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(z.real, z.imag, 'b-', lw=3, label="Original")
plt.plot(z_rec.real, z_rec.imag, 'r--', lw=3, label="Reconstruit (amélioré)")
plt.legend(); plt.axis('equal'); plt.grid(True, alpha=0.3)
plt.title("Apprentissage global du contour de face2.png")

plt.subplot(1, 2, 2)
plt.plot(z.real[:400], z.imag[:400], 'b-', lw=2)
plt.plot(z_rec.real[:400], z_rec.imag[:400], 'r--', lw=2)
plt.axis('equal'); plt.title("Zoom sur début du tracé")
plt.tight_layout()
plt.show()

# ===============================
# 8. BRAS ROBOTISÉ (inchangé)
# ===============================
link_lengths = [40, 35, 30, 25, 20, 15]
total_height = sum(link_lengths)

def ik_6dof_with_z(tx, ty):
    cx, cy, cz = 0.0, 0.0, total_height
    positions = [np.array([cx, cy, cz])]
    for L in link_lengths:
        dx, dy = tx - cx, ty - cy
        dist = np.hypot(dx, dy)
        angle = np.arctan2(dy, dx) if dist > 1e-6 else 0.0
        step = min(L, dist)
        cx += step * np.cos(angle)
        cy += step * np.sin(angle)
        cz -= L
        positions.append(np.array([cx, cy, cz]))
    return positions

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-140, 140); ax.set_ylim(-140, 140); ax.set_zlim(0, total_height + 20)
ax.view_init(elev=30, azim=-70)
ax.set_title("Bras robotisé - Tracé appris de face2.png")

arm_lines = [ax.plot([], [], [], 'b-', lw=6)[0] for _ in range(len(link_lengths))]
joint_lines = [ax.plot([], [], [], 'ro', ms=8)[0] for _ in range(len(link_lengths) + 1)]
drawing_line, = ax.plot([], [], [], 'k-', lw=3, alpha=0.9)
trail = []

def animate(i):
    idx = i % len(z_rec)
    Z = z_rec[idx]
    tx, ty = Z.real, Z.imag
    positions = ik_6dof_with_z(tx, ty)
    pos_array = np.array(positions)
    for j, line in enumerate(arm_lines):
        p0, p1 = pos_array[j], pos_array[j+1]
        line.set_data([p0[0], p1[0]], [p0[1], p1[1]])
        line.set_3d_properties([p0[2], p1[2]])
    for j, line in enumerate(joint_lines):
        p = pos_array[j]
        line.set_data([p[0]], [p[1]])
        line.set_3d_properties([p[2]])
    trail.append(complex(tx, ty))
    if len(trail) > 1500: trail.pop(0)
    txs = [p.real for p in trail]
    tys = [p.imag for p in trail]
    drawing_line.set_data(txs, tys)
    drawing_line.set_3d_properties([0] * len(txs))
    return arm_lines + joint_lines + [drawing_line]

print("Animation...")
anim = FuncAnimation(fig, animate, frames=len(z_rec), interval=15, blit=False, repeat=True)
plt.tight_layout()
plt.show()
