# =============================================================================
# 1-laplace-drawing-recurrent.py
# Proof-of-Concept : LSTM + reconstruction de Fourier dans le domaine de Laplace
# Auteur : Eric Marchand
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from skimage import io, color, measure
from scipy import ndimage
import torch
import torch.nn as nn

# -------------------------------------------------
# 1. Chargement et pré-traitement de l’image
# -------------------------------------------------
print("Loading and preprocessing image...")
img = io.imread("face.png")                     # contour noir sur fond blanc
if img.shape[-1] == 4:
    img = img[..., :3]

gray = color.rgb2gray(img)
edges = gray < 0.5
edges = ndimage.binary_fill_holes(edges)
contours = measure.find_contours(edges, 0.8)
points = np.concatenate(contours)

x, y = points[:, 1], -points[:, 0]               # Y positif vers le haut
x -= x.mean(); y -= y.mean()
z = x + 1j * y                                   # représentation complexe

# -------------------------------------------------
# 2. Densification du contour (N = 6000 points)
# -------------------------------------------------
N = 6000
z = np.interp(np.linspace(0, len(z), N), np.arange(len(z)), z)
print(f"Contour densified to {N} points.")

# -------------------------------------------------
# 3. Calcul de la courbure et des features
# -------------------------------------------------
dx  = np.gradient(z.real)
dy  = np.gradient(z.imag)
ddx = np.gradient(dx)
ddy = np.gradient(dy)

curvature = np.abs(dx*ddy - dy*ddx) / (dx**2 + dy**2 + 1e-8)**1.5
curvature /= curvature.max() + 1e-8

features = np.stack([
    curvature,
    np.gradient(curvature),
    np.gradient(np.abs(dx + 1j*dy))
], axis=1)

target = 1 / (1 + 3*curvature)                  # vitesse cible ∈ (0,1]
target = target[:, None]

X_seq = torch.tensor(features, dtype=torch.float32)   # (N, 3)
y_seq = torch.tensor(target,   dtype=torch.float32)   # (N, 1)

# -------------------------------------------------
# 4. Modèle LSTM (RecurrentSpeedNet)
# -------------------------------------------------
class RecurrentSpeedNet(nn.Module):
    def __init__(self, input_size=3, hidden_size=16, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 8), nn.ReLU(),
            nn.Linear(8, 1), nn.Sigmoid()
        )

    def forward(self, x, hidden=None):
        """
        x : (batch, seq_len, input_size)   ou   (seq_len, input_size)
        retourne (out, hidden)
        """
        # ---- normalisation de la forme ----
        if x.dim() == 2:                     # (seq_len, feat)
            x = x.unsqueeze(0)               # (1, seq_len, feat)
        elif x.dim() == 4:                   # cas erroné (1,1,1,feat)
            x = x.squeeze(1)                 # (1, seq_len, feat)

        lstm_out, hidden = self.lstm(x, hidden)   # (B, T, H)
        out = self.fc(lstm_out)                   # (B, T, 1)
        return out, hidden

# -------------------------------------------------
# 5. Entraînement (teacher-forcing sur la séquence complète)
# -------------------------------------------------
print("Training LSTM (full-sequence teacher forcing)...")
model = RecurrentSpeedNet()
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

X_batch = X_seq.unsqueeze(0)        # (1, N, 3)
y_batch = y_seq.unsqueeze(0)        # (1, N, 1)

model.train()
for epoch in range(300):
    opt.zero_grad()
    out, _ = model(X_batch)
    loss = loss_fn(out, y_batch)
    loss.backward()
    opt.step()

    if epoch % 100 == 0 or epoch == 299:
        print(f"  Epoch {epoch:3d} | Loss: {loss.item():.6f}")

print(f"Training complete. Final loss = {loss.item():.6f}")

# -------------------------------------------------
# 6. Coefficients FFT (base de la reconstruction)
# -------------------------------------------------
c     = np.fft.fft(z) / N
freqs = np.fft.fftfreq(N, 1/N)

# -------------------------------------------------
# 7. Conversion sécurisée NumPy
                    # -------------------------------------------------
def ensure_numpy(arr):
    if isinstance(arr, np.ndarray):
        return arr
    try:
        return arr.detach().cpu().numpy()
    except Exception:
        return np.array(arr, dtype=float)

x_np = ensure_numpy(x)
y_np = ensure_numpy(y)

# -------------------------------------------------
# 8. Préparation du canvas d’animation
# -------------------------------------------------
fig, ax = plt.subplots(figsize=(7,7))
scale = 1.1
ax.set_xlim(-np.abs(x_np).max()*scale, np.abs(x_np).max()*scale)
ax.set_ylim(-np.abs(y_np).max()*scale, np.abs(y_np).max()*scale)
ax.set_aspect('equal')
ax.axis('off')

line,  = ax.plot([], [], 'k.', markersize=1.5, alpha=0.7)   # trace
point, = ax.plot([], [], 'ro', markersize=10)              # pointe courante
trail = []

# état caché du LSTM (conservé entre frames)
rnn_hidden = None

# -------------------------------------------------
# 9. Boucle d’animation : inférence en ligne + modulation Laplace
# -------------------------------------------------
def animate(frame):
    global rnn_hidden

    t = 2*np.pi*frame/N
    max_k = 800
    k_vals = np.arange(-max_k, max_k+1)
    Z = 0.0 + 0.0j

    # ---- feature courante (1 seul pas) ----
    idx = frame % N
    cur_feat = X_seq[idx:idx+1]                 # (1, 3)

    # ---- inférence LSTM en ligne ----
    model.eval()
    with torch.no_grad():
        # on passe toujours un batch de taille 1 et une séquence de longueur 1
        speed_pred, rnn_hidden = model(cur_feat.unsqueeze(0), rnn_hidden)
    accel_factor = 0.5 + 0.8*speed_pred.item()   # [0.5, 1.3]

    # ---- reconstruction de Fourier avec fréquence modulée ----
    for k in k_vals:
        i = k % N
        sigma = 0.001*np.sin(0.2*t + 0.1*k)
        omega = 2*np.pi*freqs[i]*accel_factor
        Z += c[i] * np.exp((sigma + 1j*omega)*t)

    trail.append(Z)

    # arrêt automatique
    if frame >= N-1:
        print("Drawing complete. Stopping animation.")
        anim.event_source.stop()

    line.set_data(np.real(trail), np.imag(trail))
    point.set_data([Z.real], [Z.imag])
    return line, point

# -------------------------------------------------
# 10. Lancement de l’animation
# -------------------------------------------------
print("Starting animation with recurrent speed control...")
anim = FuncAnimation(fig, animate, frames=N,
                     interval=15, blit=False, repeat=False)
plt.tight_layout()
plt.show()
