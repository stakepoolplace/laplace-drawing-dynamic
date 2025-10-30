#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Laplace Perceptron entraînable (version spectro-temporelle)
-----------------------------------------------------------
• Chaque unité = oscillateur amorti : exp(-s t) * (a sin(...) + b cos(...))
• s, ω, phase, amplitudes A/B = PARAMÈTRES APPRIS (pas des buffers figés)
• Régularisation Kuramoto optionnelle pour garder les phases cohérentes
• Démo : on reconstruit un signal amorti multi-fréquences
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

# ============================================================
# 1) Modèle
# ============================================================

class LaplacePerceptron(nn.Module):
    """
    Vrai Laplace Perceptron avec paramètres apprenables :
    - fréquences = grille + Δω appris
    - amortissements s_k >= 0 (softplus)
    - phases apprises
    - amplitudes sin/cos apprises
    - projection linéaire sur la sortie
    """
    def __init__(
        self,
        n_units=48,
        d_out=1,
        w_min=1.0,
        w_max=8.0,
        s_init=0.1,
        s_max=3.0,
        use_kuramoto_reg=True,
        kuramoto_strength=1e-3,
    ):
        super().__init__()
        self.n_units = n_units
        self.s_max = s_max
        self.use_kuramoto_reg = use_kuramoto_reg
        self.kuramoto_strength = kuramoto_strength

        # -------- fréquences : base + delta appris --------
        base_w = torch.linspace(w_min, w_max, n_units)
        self.register_buffer("w0", base_w)               # [N], non entraîné
        self.domega = nn.Parameter(torch.zeros(n_units)) # [N], entraîné

        # -------- amortissements s_k >= 0 --------
        # on part d'un s_init et on passe par softplus
        sp_inv = np.log(np.exp(s_init) - 1.0)  # inverse approx de softplus
        self.raw_s = nn.Parameter(torch.full((n_units,), float(sp_inv)))

        # -------- phases, amplitudes --------
        self.phi = nn.Parameter(torch.zeros(n_units))         # [N]
        self.a   = nn.Parameter(torch.randn(n_units) * 0.05)  # [N]
        self.b   = nn.Parameter(torch.randn(n_units) * 0.05)  # [N]

        # -------- projection finale --------
        # on donne à la linéaire sin ET cos de toutes les unités (2N)
        self.mixer = nn.Linear(2 * n_units, d_out, bias=False)

    def forward(self, t):
        """
        t : [B, T] grille temporelle
        retourne : y_pred [B, T], reg (scalaire)
        """
        B, T = t.shape
        device = t.device

        # amortissements
        s = torch.nn.functional.softplus(self.raw_s).clamp(max=self.s_max)  # [N]
        # fréquences
        w = (self.w0 + self.domega).clamp(min=1e-6)                         # [N]

        # mise en forme
        tBT    = t.unsqueeze(1)                 # [B, 1, T]
        s1N1   = s.view(1, -1, 1)               # [1, N, 1]
        w1N1   = w.view(1, -1, 1)               # [1, N, 1]
        phi1N1 = self.phi.view(1, -1, 1)        # [1, N, 1]
        a1N1   = self.a.view(1, -1, 1)          # [1, N, 1]
        b1N1   = self.b.view(1, -1, 1)          # [1, N, 1]

        # bases Laplace
        decay = torch.exp(-s1N1 * tBT)          # [B, N, T]
        phase = w1N1 * tBT + phi1N1             # [B, N, T]
        sinBNT = decay * torch.sin(phase)
        cosBNT = decay * torch.cos(phase)

        # features pour la sortie
        feats = torch.cat([sinBNT, cosBNT], dim=1).permute(0, 2, 1)  # [B, T, 2N]
        y = self.mixer(feats).squeeze(-1)                            # [B, T]

        # --------------------------------------------------
        # régularisation Kuramoto (optionnelle)
        # --------------------------------------------------
        reg = torch.tensor(0.0, device=device)
        if self.use_kuramoto_reg and self.kuramoto_strength > 0.0:
            # phase moyenne par unité
            phase_mean = phase.mean(dim=2)               # [B, N]
            diff = phase_mean.unsqueeze(2) - phase_mean.unsqueeze(1)  # [B, N, N]
            # plus la moyenne de cos(diff) est haute → plus les phases sont alignées
            order_param = torch.cos(diff).mean(dim=(1, 2))  # [B]
            # on veut les rapprocher → pénalité = -order_param
            reg = -order_param.mean() * self.kuramoto_strength

        return y, reg


# ============================================================
# 2) Données (signal cible)
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH = 1
T_total = 10.0
T_steps = 2000

# grille de temps
t = torch.linspace(0, T_total, T_steps).unsqueeze(0).repeat(BATCH, 1).to(device)  # [B, T]

# signal cible : somme amortie de sines/cos + un terme purement amorti
signal = (
    torch.exp(-0.1 * t) * torch.sin(3 * t)
    + 0.5 * torch.exp(-0.3 * t) * torch.cos(7 * t)
    + 0.2 * torch.exp(-0.5 * t)
).to(device)  # [B, T]


# ============================================================
# 3) Entraînement
# ============================================================

model = LaplacePerceptron(
    n_units=48,
    w_min=1.0,
    w_max=10.0,
    s_init=0.1,
    s_max=3.0,
    use_kuramoto_reg=True,
    kuramoto_strength=1e-3,
).to(device)

opt = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

losses = []
EPOCHS = 20000  # tu peux monter à 50000 comme avant

for epoch in trange(EPOCHS, desc="Training"):
    opt.zero_grad()
    y_pred, reg = model(t)               # [B, T], scalaire
    loss = loss_fn(y_pred, signal) + reg
    loss.backward()
    opt.step()
    losses.append(loss.item())


# ============================================================
# 4) Visualisation
# ============================================================

y = y_pred.detach().cpu().numpy().squeeze()     # [T]
t_cpu = t.detach().cpu().numpy().squeeze()      # [T]
s = signal.detach().cpu().numpy().squeeze()     # [T]

plt.figure(figsize=(10, 8))

# (1) Signal vs reconstruction
plt.subplot(3, 1, 1)
plt.plot(t_cpu, s, 'k--', label="Target")
plt.plot(t_cpu, y, 'r', label="Model")
plt.legend()
plt.title("Laplace Perceptron — time-aligned output")

# (2) Courbe de perte
plt.subplot(3, 1, 2)
plt.plot(losses)
plt.title("Training loss (MSE + Kuramoto)")
plt.xlabel("Epochs")
plt.ylabel("Loss")

# (3) Spectre fréquentiel (FFT)
fft_s = np.abs(np.fft.rfft(s))
fft_y = np.abs(np.fft.rfft(y))
freqs = np.fft.rfftfreq(len(s), d=T_total / len(s))

plt.subplot(3, 1, 3)
plt.plot(freqs, fft_s, 'k--', label="Target FFT")
plt.plot(freqs, fft_y, 'r', alpha=0.85, label="Model FFT")
plt.title("Frequency spectra")
plt.xlabel("Frequency (Hz)")
plt.legend()

plt.tight_layout()
plt.show()
