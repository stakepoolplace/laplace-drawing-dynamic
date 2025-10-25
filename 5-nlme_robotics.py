# =============================================================================
# nlme_robotics_v9.7_syncfix3.py
# NLME complet + cube filaire 3D + affichage de progression + correction shape mismatch
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

# ===============================
# 0. PARAMÈTRES
# ===============================
N = 1024
win, hop = 128, 16
n_keep = 32
latent_dim = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🧠 Running on: {device}")

# ===============================
# 1. GÉNÉRATION DU CUBE 3D
# ===============================
def generate_cube_path(N):
    s = 70
    z_offset = 80
    vertices = np.array([
        [-s, -s, -s + z_offset],
        [ s, -s, -s + z_offset],
        [ s,  s, -s + z_offset],
        [-s,  s, -s + z_offset],
        [-s, -s,  s + z_offset],
        [ s, -s,  s + z_offset],
        [ s,  s,  s + z_offset],
        [-s,  s,  s + z_offset]
    ])
    path_order = [0, 1, 2, 3, 0, 4, 5, 1, 2, 6, 7, 3, 0]
    pts = []
    for k in range(len(path_order) - 1):
        p1, p2 = vertices[path_order[k]], vertices[path_order[k + 1]]
        seg = np.linspace(p1, p2, N // (len(path_order) - 1))
        pts.append(seg)
    pts = np.vstack(pts)
    theta = np.pi / 6
    rot = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    pts = pts @ rot.T
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    return pts, vertices, edges

points, vertices, edges = generate_cube_path(N)
x, y, z = points[:,0], points[:,1], points[:,2]
z_complex = x + 1j * y
v = np.gradient(z_complex)
a = np.gradient(v)
print("✅ Cube 3D généré")

# ===============================
# 2. STFT GPU
# ===============================
def stft_features(sig, win=128, hop=16, n_keep=32):
    sig_t = torch.tensor(sig, dtype=torch.float32, device=device)
    window = torch.hann_window(win, device=device)
    stft_res = torch.stft(sig_t, n_fft=win, hop_length=hop,
                          window=window, return_complex=True)
    amp = stft_res.abs()[:n_keep].T
    phase = torch.angle(stft_res[:n_keep].T)
    dphi = torch.diff(phase, dim=0)
    dphi = (dphi + np.pi) % (2*np.pi) - np.pi
    phase_unwrapped = torch.cat([phase[:1], phase[:1] + torch.cumsum(dphi, dim=0)], dim=0)
    feats = torch.cat([amp, phase_unwrapped], dim=1)
    return feats.cpu().numpy()

feat_x = stft_features(x)
feat_y = stft_features(y)
feat_z = stft_features(z)
L = min(len(feat_x), len(feat_y), len(feat_z))
X_stft = np.stack([feat_x[:L], feat_y[:L], feat_z[:L]], axis=1)

# Positional encoding
pos = np.arange(L)[:, None]
pe = np.zeros((L, 64))
pe[:, 0::2] = np.sin(pos / 10000 ** (2 * np.arange(32) / 64))
pe[:, 1::2] = np.cos(pos / 10000 ** (2 * np.arange(32) / 64))
pe = np.repeat(pe[:, None, :], 3, axis=1)
X_raw = np.concatenate([X_stft, pe], axis=2)
X_flat = X_raw.reshape(X_raw.shape[0], -1)
print(f"✅ X_raw shape: {X_raw.shape}")

# ===============================
# 3. MODÈLE NLME
# ===============================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1), :]

class NLME(nn.Module):
    def __init__(self, input_dim=384, d_model=128, nhead=8, num_layers=3,
                 latent_dim=32, seq_len=129):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, 256, 0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_logvar = nn.Linear(d_model, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, seq_len * 2 * n_keep * 2)
        )
        self.seq_len = seq_len
    def encode(self, x):
        x = self.pos_encoder(self.proj(x))
        x = self.encoder(x)
        mu, logvar = self.fc_mu(x.mean(dim=1)), self.fc_logvar(x.mean(dim=1))
        return mu, logvar
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def decode(self, z): return self.decoder(z).view(1, self.seq_len, 2 * n_keep * 2)
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        spec_pred = self.decode(z)
        return z, spec_pred, mu, logvar

# ===============================
# 4. ENTRAÎNEMENT
# ===============================
X_tensor = torch.tensor(X_flat, dtype=torch.float32).unsqueeze(0).to(device)
y_spec = torch.tensor(np.stack([feat_x[:L], feat_y[:L]], axis=1)
                      .reshape(L, -1), dtype=torch.float32).unsqueeze(0).to(device)
model = NLME(seq_len=L).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

print("🚀 Entraînement du NLME...")
for epoch in range(400):
    opt.zero_grad()
    z, spec_pred, mu, logvar = model(X_tensor)
    loss_recon = loss_fn(spec_pred, y_spec)
    loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = loss_recon + 0.01 * loss_kl
    loss.backward(); opt.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f}")

# ===============================
# 5. DÉCODAGE + SYNC FIX
# ===============================
model.eval()
with torch.no_grad():
    _, spec_pred, _, _ = model(X_tensor)
    spec_pred = spec_pred.cpu().numpy().reshape(L, 2, 2*n_keep)

for i in range(1, len(spec_pred)):
    dphase = spec_pred[i, :, n_keep:] - spec_pred[i-1, :, n_keep:]
    dphase = (dphase + np.pi) % (2*np.pi) - np.pi
    spec_pred[i, :, n_keep:] = spec_pred[i-1, :, n_keep:] + dphase

reconstructed = np.zeros(N, dtype=complex)
count = np.zeros(N)
window = np.hanning(win)
for i in range(L):
    start = i * hop
    amp_x, phase_x = spec_pred[i,0,:n_keep], spec_pred[i,0,n_keep:]
    amp_y, phase_y = spec_pred[i,1,:n_keep], spec_pred[i,1,n_keep:]
    spec_x = np.zeros(win, dtype=complex)
    spec_y = np.zeros(win, dtype=complex)
    spec_x[:n_keep] = amp_x * np.exp(1j * phase_x)
    spec_y[:n_keep] = amp_y * np.exp(1j * phase_y)
    spec_x[-n_keep+1:] = np.conj(spec_x[1:n_keep])
    spec_y[-n_keep+1:] = np.conj(spec_y[1:n_keep])
    seg_x = np.fft.ifft(spec_x) * window
    seg_y = np.fft.ifft(spec_y) * window
    end = min(start + win, N)
    seg = seg_x[:end-start] + 1j * seg_y[:end-start]
    reconstructed[start:end] += seg
    count[start:end] += window[:end-start]
count[count == 0] = 1
z_rec = reconstructed / count
z_rec -= np.mean(z_rec)
z_rec = np.convolve(z_rec, np.ones(5)/5, mode='same')
z_rec = z_rec / np.max(np.abs(z_rec)) * 120.0

min_len = min(len(z_rec), len(points))
z_rec = z_rec[:min_len]
z_depth = points[:min_len, 2]
print("✅ Synchronisation effectuée")

# ===============================
# 6. BRAS ROBOTISÉ + VISUALISATION
# ===============================
link_lengths = [40, 35, 30, 25, 20, 15]
total_height = sum(link_lengths)
max_reach = total_height * 0.9

def ik_6dof_with_z(tx, ty, tz):
    cx, cy, cz = 0.0, 0.0, total_height
    positions = [np.array([cx, cy, cz])]
    for L in link_lengths:
        dx, dy, dz = tx - cx, ty - cy, tz - cz
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        if dist < 1e-6:
            positions.append(np.array([tx, ty, tz]))
            while len(positions) < len(link_lengths) + 1:
                positions.append(positions[-1].copy())
            break
        step = min(L, dist)
        ratio = step / dist
        cx += dx * ratio; cy += dy * ratio; cz += dz * ratio
        positions.append(np.array([cx, cy, cz]))
    while len(positions) < len(link_lengths) + 1:
        positions.append(positions[-1].copy())
    return positions

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-140, 140); ax.set_ylim(-140, 140); ax.set_zlim(0, total_height + 140)
ax.view_init(elev=25, azim=-60)
ax.set_title("🤖 NLME v9.7-syncfix3 : cube filaire + trajectoire reconstruite")

# ✅ Sécurisation des shapes
x = np.ravel(np.array(x.detach().cpu().numpy() if torch.is_tensor(x) else x))
y = np.ravel(np.array(y.detach().cpu().numpy() if torch.is_tensor(y) else y))
z = np.ravel(np.array(z.detach().cpu().numpy() if torch.is_tensor(z) else z))
min_len = min(len(x), len(y), len(z), min_len)

# cube filaire + trajectoire originale
for i, j in edges:
    p1, p2 = vertices[i], vertices[j]
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='gray', alpha=0.3, lw=2, ls='--')
ax.plot(x[:min_len], y[:min_len], z[:min_len], color='orange', lw=1.5, alpha=0.4, label="original")

arm_lines = [ax.plot([], [], [], 'b-', lw=6)[0] for _ in link_lengths]
joint_lines = [ax.plot([], [], [], 'ro', ms=8)[0] for _ in range(len(link_lengths)+1)]
trail_points, = ax.plot([], [], [], '-', lw=4, alpha=0.9, color='cyan', label="NLME reconstruit")
trail = []

def animate(i):
    idx = i % len(z_rec)
    if idx % 100 == 0:
        pct = 100 * idx / len(z_rec)
        print(f"🌀 Frame {idx}/{len(z_rec)}  ({pct:.1f}%)")
    Z = z_rec[idx]; tx, ty = Z.real, Z.imag; tz = z_depth[idx]
    dist = np.sqrt(tx**2 + ty**2 + tz**2)
    if dist > max_reach:
        s = max_reach / dist; tx, ty, tz = tx*s, ty*s, tz*s
    arr = np.array(ik_6dof_with_z(tx, ty, tz))
    for j, line in enumerate(arm_lines):
        p0, p1 = arr[j], arr[j+1]
        line.set_data([p0[0], p1[0]], [p0[1], p1[1]])
        line.set_3d_properties([p0[2], p1[2]])
    for j, line in enumerate(joint_lines):
        p = arr[j]; line.set_data([p[0]], [p[1]]); line.set_3d_properties([p[2]])
    trail.append([tx, ty, tz])
    if len(trail) > 1300: trail.pop(0)
    trail_np = np.array(trail)
    trail_points.set_data(trail_np[:,0], trail_np[:,1])
    trail_points.set_3d_properties(trail_np[:,2])
    ax.view_init(elev=25+5*np.sin(i/150), azim=-60+i/2)
    return arm_lines + joint_lines + [trail_points]

print("🎨 Animation...")
anim = FuncAnimation(fig, animate, frames=len(z_rec), interval=20, blit=False, repeat=True)
ax.legend()
plt.tight_layout(); plt.show()

print("\n✅ NLME v9.7-syncfix3 : stable, aligné, formes 3D corrigées.")
