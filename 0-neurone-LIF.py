import numpy as np
import matplotlib.pyplot as plt

# 1. Temps et pas de simulation
dt = 0.001
T = 2.0
t = np.arange(0, T, dt)

# 2. Signal d'entrée : train d'impulsions de plus en plus rapprochées
u = np.zeros_like(t)
pulse_starts = [0.2, 0.6, 0.9, 1.1, 1.25, 1.35, 1.42, 1.48, 1.52, 1.56, 1.59, 1.62, 1.61]  # impulsions rapprochées à la fin
pulse_width = 0.04
for start in pulse_starts:
    u[(t >= start) & (t < start + pulse_width)] = 1.0

# 3. Paramètres du neurone LIF
tau = 0.05          # constante de temps
Vth = 0.6           # seuil de déclenchement
Vreset = 0.0        # reset
V = 0.0
V_trace = []
spikes = []

# 4. Simulation
for I in u:
    dV = (-V + I) * dt / tau
    V += dV
    if V >= Vth:
        spikes.append(1)
        V = Vreset
    else:
        spikes.append(0)
    V_trace.append(V)

# 5. Affichage
plt.figure(figsize=(10,5))

plt.subplot(3,1,1)
plt.plot(t, u, 'k', linewidth=1.2)
plt.title("Signal d'entrée : impulsions de plus en plus rapprochées")
plt.ylabel("u(t)")

plt.subplot(3,1,2)
plt.plot(t, V_trace, 'b')
plt.axhline(Vth, color='r', linestyle='--', label='Seuil')
plt.title("Potentiel de membrane (intégration et fuite)")
plt.ylabel("V(t)")
plt.legend()

plt.subplot(3,1,3)
plt.plot(t, np.array(spikes), 'r|', markersize=20)
plt.title("Spikes émis (sortie du neurone)")
plt.ylabel("Spike")
plt.xlabel("Temps (s)")

plt.tight_layout()
plt.show()

