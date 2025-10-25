#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation d’un petit réseau de neurones "Laplace"
---------------------------------------------------
Chaque neurone encode le passé selon une exponentielle e^{-s_i t}.
Une couche inverse combine ces sorties pour reconstruire le signal original.
"""

import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. Signal d’entrée (stimulus)
# -------------------------------
t = np.linspace(0, 5, 1000)  # temps
f = np.exp(-((t - 2.5)**2) / 0.05)  # un flash à t=2.5 (impulsion)
# f = np.sin(2*np.pi*t*1.5) * (t < 3)  # (autre essai : onde sinusoïdale)

# -------------------------------
# 2. Paramètres des neurones Laplace
# -------------------------------
s_values = np.array([0.2, 0.5, 1.0, 2.0, 5.0])  # constantes de décroissance
n_neurons = len(s_values)

# -------------------------------
# 3. Encodage Laplace (intégration exponentielle)
# -------------------------------
F = np.zeros((n_neurons, len(t)))

for i, s in enumerate(s_values):
    kernel = np.exp(-s * (t[:, None] - t[None, :])) * (t[:, None] >= t[None, :])
    # Convolution exponentielle (∫ f(τ)e^{-s(t-τ)} dτ)
    F[i, :] = np.trapz(f[None, :] * kernel, t, axis=1)

# -------------------------------
# 4. Décodage (Laplace inverse approchée)
# -------------------------------
# On cherche des poids w_i qui reconstruisent f à partir des F(s_i)
# Résolution linéaire : f ≈ Σ_i w_i F_i
w, _, _, _ = np.linalg.lstsq(F.T, f, rcond=None)
f_recon = np.dot(w, F)

# -------------------------------
# 5. Visualisation
# -------------------------------
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# Signal original
axs[0].plot(t, f, 'k', lw=2, label='Signal original f(t)')
axs[0].set_title("Signal d’entrée (ex: stimulus sensoriel)")
axs[0].legend()

# Activité des neurones Laplace
for i, s in enumerate(s_values):
    axs[1].plot(t, F[i], label=f"Neurone {i+1} (s={s})")
axs[1].set_title("Activité des neurones Laplace (encodage exponentiel)")
axs[1].legend(loc='upper right', fontsize=8)

# Reconstruction (Laplace inverse)
axs[2].plot(t, f, 'k--', lw=1.5, label='Original')
axs[2].plot(t, f_recon, 'r', lw=2, label='Reconstruction (Laplace inverse)')
axs[2].set_title("Reconstruction du signal par Laplace inverse (pondération neuronale)")
axs[2].legend()

plt.tight_layout()
plt.show()

