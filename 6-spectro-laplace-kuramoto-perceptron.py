#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spectro-Laplace Perceptron synchronisé — version shapes robustes
----------------------------------------------------------------
• Sortie temporelle alignée : [B, T]
• Synchronisation Kuramoto entre unités (cascade)
• Visualisation : signal, perte, FFT
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

class SpectroLaplacePerceptron(nn.Module):
    def __init__(self, n_units=12, s_range=(0.1, 3.0), w_range=(1.0, 8.0), K_phase=2.0):
        super().__init__()
        self.n_units = n_units
        self.K_phase = K_phase

        # Paramètres spectro-temporels (non entraînés ici)
        self.register_buffer("s", torch.linspace(*s_range, n_units))  # [N]
        self.register_buffer("w", torch.linspace(*w_range, n_units))  # [N]

        # Projection temps-distribuée: 2N -> 1
        self.linear = nn.Linear(2 * n_units, 1)

    def forward(self, t, phi_prev):
        """
        t        : [B, T]  (grille de temps par batch)
        phi_prev : [B, N]  (phases précédentes)
        Retourne : y_pred [B, T], phi_new [B, N]
        """
        B, T = t.shape
        N    = self.n_units

        # Mise en forme explicite pour broadcasting sûr
        # tBT : [B, 1, T], s1N1 : [1, N, 1], w1N1 : [1, N, 1], phiBN1 : [B, N, 1]
        tBT   = t.unsqueeze(1)                          # [B, 1, T]
        s1N1  = self.s.view(1, N, 1)                    # [1, N, 1]
        w1N1  = self.w.view(1, N, 1)                    # [1, N, 1]
        phiBN1= phi_prev.view(B, N, 1)                  # [B, N, 1]

        # Bases spectro-temporelles amorties
        exp_decayBNT = torch.exp(-s1N1 * tBT)           # [B, N, T]
        osc_phaseBNT = w1N1 * tBT + phiBN1              # [B, N, T]

        sinBNT = exp_decayBNT * torch.sin(osc_phaseBNT) # [B, N, T]
        cosBNT = exp_decayBNT * torch.cos(osc_phaseBNT) # [B, N, T]

        # --- Synchronisation Kuramoto (sur la phase moyenne temporelle) ---
        phase_meanBN = osc_phaseBNT.mean(dim=2)         # [B, N]
        diffBNN = phase_meanBN.unsqueeze(2) - phase_meanBN.unsqueeze(1)  # [B, N, N]
        sync_termBN = torch.sin(diffBNN).mean(dim=2)    # [B, N]
        phi_new = phi_prev + self.K_phase * sync_termBN * 0.01

        # --- Projection "time-distributed" ---
        # Concat features sur l'axe neurones -> [B, 2N, T], puis permuter en [B, T, 2N]
        featsB2NT = torch.cat([sinBNT, cosBNT], dim=1)  # [B, 2N, T]
        featsBT2N = featsB2NT.permute(0, 2, 1)          # [B, T, 2N]

        # Linear appliquée à chaque pas de temps → [B, T, 1] puis squeeze → [B, T]
        y_pred = torch.tanh(self.linear(featsBT2N)).squeeze(-1)  # [B, T]

        return y_pred, phi_new


# ============================================================
# 2) Données (signal cible)
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH = 1
T_total = 10.0
T_steps = 2000

t = torch.linspace(0, T_total, T_steps).unsqueeze(0).repeat(BATCH, 1).to(device)  # [B, T]

# Signal amorti complexe: plusieurs fréquences + décroissances
signal = (
    torch.exp(-0.1*t)*torch.sin(3*t)
    + 0.5*torch.exp(-0.3*t)*torch.cos(7*t)
    + 0.2*torch.exp(-0.5*t)
).to(device)  # [B, T]


# ============================================================
# 3) Entraînement
# ============================================================

model = SpectroLaplacePerceptron(n_units=48).to(device)
opt = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

phi = torch.zeros(BATCH, model.n_units, device=device)  # [B, N]

losses = []
for epoch in trange(50000, desc="Training"):
    opt.zero_grad()
    y_pred, phi = model(t, phi)                  # [B, T], [B, N]
    loss = loss_fn(y_pred, signal)               # shapes alignées
    loss.backward()
    opt.step()
    losses.append(loss.item())


# ============================================================
# 4) Visualisation
# ============================================================

y = y_pred.detach().cpu().numpy().squeeze()     # [T]
t_cpu = t.detach().cpu().numpy().squeeze()      # [T]
s = signal.detach().cpu().numpy().squeeze()     # [T]

plt.figure(figsize=(10,8))

# (1) Signal vs reconstruction
plt.subplot(3,1,1)
plt.plot(t_cpu, s, 'k--', label="Target")
plt.plot(t_cpu, y, 'r', label="Model")
plt.legend()
plt.title("Spectro-Laplace Perceptron — time-aligned output")

# (2) Courbe de perte
plt.subplot(3,1,2)
plt.plot(losses)
plt.title("Training loss (MSE)")
plt.xlabel("Epochs")
plt.ylabel("Loss")

# (3) Spectre fréquentiel (FFT)
fft_s = np.abs(np.fft.rfft(s))
fft_y = np.abs(np.fft.rfft(y))
freqs = np.fft.rfftfreq(len(s), d=T_total/len(s))

plt.subplot(3,1,3)
plt.plot(freqs, fft_s, 'k--', label="Target FFT")
plt.plot(freqs, fft_y, 'r', alpha=0.85, label="Model FFT")
plt.title("Frequency spectra")
plt.xlabel("Frequency (Hz)")
plt.legend()

plt.tight_layout()
plt.show()
