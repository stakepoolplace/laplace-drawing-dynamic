import numpy as np
import matplotlib.pyplot as plt

# 1. Signal d'entrée : train d'impulsions
dt = 0.001
T = 3.0
t = np.arange(0, T, dt)
u = np.zeros_like(t)

# Création d'un train d'impulsions espacées régulièrement
pulse_interval = 0.5  # secondes entre impulsions
pulse_width = 0.05    # durée de chaque impulsion
for start in np.arange(0.2, T, pulse_interval):
    u[(t >= start) & (t < start + pulse_width)] = 1.0

# 2. Réseau d'intégrateurs exponentiels (modèle Laplace)
N = 6
taus = np.logspace(-2, 0, N)  # constantes de temps de 0.01 à 1 s
x = np.zeros((N, len(t)))

for i in range(1, len(t)):
    dx = (-x[:, i-1] + u[i]) * dt / taus
    x[:, i] = x[:, i-1] + dx

# 3. Reconstruction (approximation inverse de Laplace)
weights = np.exp(-taus)
y = np.dot(weights, x)

# 4. Visualisation
plt.figure(figsize=(10,6))

plt.subplot(3,1,1)
plt.plot(t, u, 'k', linewidth=1.5)
plt.title("Signal d'entrée : train d'impulsions successives")
plt.ylabel("u(t)")

plt.subplot(3,1,2)
for i in range(N):
    plt.plot(t, x[i], label=f'τ={taus[i]:.2f}s')
plt.title("Réponse des intégrateurs neuronaux (mémoire de Laplace)")
plt.ylabel("x_i(t)")
plt.legend()

plt.subplot(3,1,3)
plt.plot(t, y, 'r', linewidth=1.5, label='Reconstruction')
plt.title("Combinaison pondérée (approximation inverse de Laplace)")
plt.ylabel("y(t)")
plt.xlabel("Temps (s)")
plt.legend()

plt.tight_layout()
plt.show()

