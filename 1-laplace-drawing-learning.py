import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from skimage import io, color, measure
from scipy import ndimage
import torch
import torch.nn as nn

# --- 1. Charger et préparer l'image ---
image = io.imread("face.png")  # ton image noir sur blanc
if image.shape[-1] == 4:
    image = image[..., :3]

gray = color.rgb2gray(image)
edges = gray < 0.5
edges = ndimage.binary_fill_holes(edges)
contours = measure.find_contours(edges, 0.8)
points = np.concatenate(contours)
x, y = points[:, 1], -points[:, 0]
x -= np.mean(x)
y -= np.mean(y)
z = x + 1j * y

# --- 2. Densification du contour ---
z = np.interp(np.linspace(0, len(z), 6000), np.arange(len(z)), z)
N = len(z)

# --- 3. Calcul de la courbure locale ---
dx = np.gradient(np.real(z))
dy = np.gradient(np.imag(z))
ddx = np.gradient(dx)
ddy = np.gradient(dy)
curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-8)**1.5
curvature = curvature / (np.max(curvature) + 1e-8)

features = np.stack([curvature, np.gradient(curvature), np.gradient(np.abs(dx + 1j * dy))], axis=1)
target = 1 / (1 + 3 * curvature)  # vitesse idéale : ralenti dans les zones de forte courbure

X = torch.tensor(features, dtype=torch.float32)
y_t = torch.tensor(target[:, None], dtype=torch.float32)

# --- 4. Réseau simple ---
class SpeedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

model = SpeedNet()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# --- 5. Entraînement court ---
for epoch in range(300):
    opt.zero_grad()
    out = model(X)
    loss = loss_fn(out, y_t)
    loss.backward()
    opt.step()
print(f"✅ Entraînement terminé. Loss finale = {loss.item():.5f}")

# --- 6. FFT ---
c = np.fft.fft(z) / N
freqs = np.fft.fftfreq(N, 1 / N)

# --- 7. Conversion sécurisée NumPy ---
def ensure_numpy(arr):
    """Convertit tout tensor PyTorch en ndarray NumPy."""
    if isinstance(arr, np.ndarray):
        return arr
    try:
        return arr.detach().cpu().numpy()
    except Exception:
        return np.array(arr, dtype=float)

x = ensure_numpy(x)
y = ensure_numpy(y)

# --- 8. Préparation du tracé ---
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-np.max(np.abs(x)) * 1.1, np.max(np.abs(x)) * 1.1)
ax.set_ylim(-np.max(np.abs(y)) * 1.1, np.max(np.abs(y)) * 1.1)
ax.set_aspect('equal')
ax.axis('off')

line, = ax.plot([], [], 'k.', markersize=2)  # Points de dessin
point, = ax.plot([], [], 'ro', markersize=12)  # Point courant (curseur)
trail = []

# --- 9. Animation avec vitesse apprise et arrêt automatique ---
def animate(frame):
    # t progresse de 0 à 2π pour un cycle complet
    t = 2 * np.pi * frame / N
    max_k = 800
    k_vals = np.arange(-max_k, max_k)
    Z = 0

    # Index borné dans [0, len(X)-1]
    idx_feat = max(0, min(int((frame / N) * len(X) - 1), len(X) - 1))

    # Facteur de vitesse prédit par le réseau
    with torch.no_grad():
        accel_factor = float(model(X[idx_feat:idx_feat + 1]).item())
    accel_factor = 0.5 + 0.8 * accel_factor  # remet dans la plage 0.5–1.3

    # Construction du point actuel avec modulation dynamique
    for k in k_vals:
        idx = k % N
        sigma = 0.001 * np.sin(0.2 * t + 0.1 * k)
        omega = 2 * np.pi * freqs[idx] * accel_factor
        Z += c[idx] * np.exp((sigma + 1j * omega) * t)

    trail.append(Z)

    # ✅ Arrêt automatique une fois le dessin complet
    if frame >= N - 1:
        print("✔️  Dessin terminé, arrêt de l'animation.")
        anim.event_source.stop()

    line.set_data(np.real(trail), np.imag(trail))
    point.set_data([np.real(Z)], [np.imag(Z)])
    return line, point

# --- 10. Lancer l'animation (sans blit ni répétition pour stabilité) ---
anim = FuncAnimation(fig, animate, frames=N, interval=15, blit=False, repeat=False)
plt.show()
