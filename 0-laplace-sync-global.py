#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mini réseau de neurones Laplace couplés (animation synchronisation)
Chaque neurone intègre exponentiellement un signal d'entrée,
leurs phases s'alignent progressivement (Kuramoto).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ----------------------------
# 1. Paramètres généraux
# ----------------------------
dt = 0.002
T  = 5.0
t  = np.arange(0, T, dt)
N  = 8
s_values = np.linspace(0.3, 4.0, N)
omega = 2*np.pi*np.linspace(4, 8, N)
K_phase = 3.0
K_feed  = 0.5

# ----------------------------
# 2. Signal d'entrée
# ----------------------------
f = np.exp(-((t-2.5)**2)/0.05)  # flash à t=2.5 s

# ----------------------------
# 3. Variables dynamiques
# ----------------------------
F = np.zeros((N, len(t)))
phi = np.random.rand(N)*2*np.pi

# ----------------------------
# 4. Simulation (pré-calcul)
# ----------------------------
for k in range(1, len(t)):
    # Couplage de phase Kuramoto
    dphi = omega + (K_phase/N)*np.sum(np.sin(phi - phi[:,None]), axis=1)
    phi += dphi * dt

    # Encodage Laplace + cohérence de phase
    dF = -s_values*F[:,k-1] + f[k]
    phase_diff = np.exp(1j*phi)[:, None] * np.exp(-1j*phi[None, :])
    coupling = K_feed * phase_diff.real.mean(axis=1)
    F[:,k] = F[:,k-1] + (dF + coupling*0.05)*dt

# Reconstruction du signal
w, *_ = np.linalg.lstsq(F.T, f, rcond=None)
f_recon = np.dot(w, F)

# ----------------------------
# 5. Animation
# ----------------------------
fig = plt.figure(figsize=(8, 6))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, :])
ax3 = fig.add_subplot(gs[2, 0])
ax4 = fig.add_subplot(gs[2, 1], polar=True)

# (1) Signal original + reconstruction
ax1.plot(t, f, 'k', lw=2, label='Signal original')
line_recon, = ax1.plot([], [], 'r', lw=1.5, label='Reconstruction')
ax1.legend()
ax1.set_xlim(0, T)
ax1.set_ylim(-0.2, 1.2)

# (2) Activité Laplace
lines_F = [ax2.plot([], [], lw=1.5, label=f"s={s_values[i]:.2f}")[0] for i in range(N)]
ax2.legend(fontsize=7, loc='upper right')
ax2.set_xlim(0, T)
ax2.set_ylim(np.min(F)*1.1, np.max(F)*1.1)
ax2.set_title("Activité des neurones Laplace")

# (3) Phases synchronisées
ax3.set_xlim(0, T)
ax3.set_ylim(-1.2, 1.2)
lines_phase = [ax3.plot([], [], lw=1)[0] for _ in range(N)]
ax3.set_title("Oscillations sin(phase_i)")

# (4) Disque de synchronisation
points, = ax4.plot([], [], 'o', color='deepskyblue', markersize=8)
ax4.set_ylim(0, 1.2)
ax4.set_yticks([])
ax4.set_title("Synchronisation de phase", pad=20)

# Animation update
def update(frame):
    idx = frame
    # reconstruction
    line_recon.set_data(t[:idx], f_recon[:idx])

    # activations F
    for i in range(N):
        lines_F[i].set_data(t[:idx], F[i, :idx])

    # phases sinusoïdales
    for i in range(N):
        lines_phase[i].set_data(t[:idx], np.sin(2*np.pi*omega[i]*t[:idx]/(2*np.pi)))

    # disque de phases
    theta = np.angle(np.exp(1j*phi))
    x = np.cos(theta)
    y = np.sin(theta)
    points.set_data(theta, np.ones_like(theta))

    return [line_recon, *lines_F, *lines_phase, points]

ani = FuncAnimation(fig, update, frames=len(t), interval=30, blit=True)
plt.tight_layout()
plt.show()

