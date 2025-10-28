#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
======================================================================
RÉSEAUX D'OSCILLATEURS vs PERCEPTRONS CLASSIQUES
======================================================================

Architecture :
- Chaque "neurone oscillateur" = oscillateur Laplacien paramétré
- Organisés en couches comme un réseau feed-forward
- Comparaison avec MLP classique (ReLU/tanh)

Tests sur plusieurs tâches :
1. Reconstruction de signaux (sinusoïdes, chirps, AM/FM)
2. Classification de séquences temporelles
3. Prédiction de séries temporelles
4. Apprentissage de fonctions périodiques
5. Détection de patterns oscillatoires

Objectif : Déterminer quand les oscillateurs sont meilleurs que les perceptrons
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import trange
from typing import List, Tuple, Dict

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cpu")

# ══════════════════════════════════════════════════════════════════
# ARCHITECTURE : NEURONE OSCILLATEUR
# ══════════════════════════════════════════════════════════════════

class OscillatorNeuron(nn.Module):
    """
    Neurone oscillateur : y(t) = e^{-s t}[ a sin(ωt) + b cos(ωt) ]
    - a, b : amplitudes
    - s : amortissement
    - ω : pulsation (peut être fixe ou apprenable)
    """
    def __init__(self, frequency, learnable_freq=False, init_damping=0.25):
        super().__init__()
        
        # Fréquence (fixe ou apprenable)
        if learnable_freq:
            self.omega = nn.Parameter(torch.tensor(2*np.pi*frequency, dtype=torch.float32))
        else:
            self.register_buffer('omega', torch.tensor(2*np.pi*frequency, dtype=torch.float32))
        
        # Amplitudes sin/cos
        self.a = nn.Parameter(torch.randn(1) * 0.1)
        self.b = nn.Parameter(torch.randn(1) * 0.1)
        
        # Amortissement (via softplus pour positivité)
        inv_softplus = np.log(np.expm1(init_damping))
        self.log_s = nn.Parameter(torch.tensor(inv_softplus, dtype=torch.float32))
    
    def forward(self, t):
        """
        t : [batch_size, seq_len] - temps
        returns : [batch_size, seq_len] - sortie oscillateur
        """
        s = torch.nn.functional.softplus(self.log_s).clamp_max(1.0)
        
        phase = self.omega * t
        exp_term = torch.exp(-s * t)
        
        return exp_term * (self.a * torch.sin(phase) + self.b * torch.cos(phase))


class OscillatorLayer(nn.Module):
    """Couche d'oscillateurs parallèles (fréquences réparties)"""
    def __init__(self, n_oscillators: int, fmin: float, fmax: float, learnable_freq=False):
        super().__init__()
        freqs = np.linspace(fmin, fmax, n_oscillators)
        self.neurons = nn.ModuleList([
            OscillatorNeuron(f, learnable_freq=learnable_freq) for f in freqs
        ])
    
    def forward(self, t):
        # t : [batch, seq]
        outputs = [neuron(t) for neuron in self.neurons]  # liste de [batch, seq]
        return torch.stack(outputs, dim=-1)  # [batch, seq, n_osc]


class DeepOscillatorNetwork(nn.Module):
    """
    Réseau profond d'oscillateurs (feed-forward)
    
    Architecture :
        Input (temps) → OscLayer1 → Linear → OscLayer2 → Linear → Output
    """
    def __init__(self, layer_sizes: List[Tuple[int, float, float]], learnable_freq=False):
        """
        layer_sizes : [(n_osc, f_min, f_max), ...]
        Exemple : [(32, 10, 100), (64, 100, 1000), (32, 1000, 5000)]
        """
        super().__init__()
        
        self.layers = nn.ModuleList([
            OscillatorLayer(n, fmin, fmax, learnable_freq=learnable_freq)
            for (n, fmin, fmax) in layer_sizes
        ])
        
        self.projections = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            in_size = layer_sizes[i][0]
            out_size = layer_sizes[i+1][0]
            self.projections.append(nn.Linear(in_size, out_size))
        
        last_size = layer_sizes[-1][0]
        self.output_layer = nn.Linear(last_size, 1)
    
    def forward(self, t):
        """
        t : [batch_size, seq_len]
        returns : [batch_size, seq_len]
        """
        # Robustify input shape: accept 1D [batch] by adding a time dim
        if t.dim() == 1:
            t = t.unsqueeze(1)
        
        # Première couche d'oscillateurs
        x = self.layers[0](t)  # [batch, seq, n_osc]
        
        # Couches intermédiaires
        for i in range(len(self.projections)):
            # Moyenne temporelle pour passer à la couche suivante
            x = x.mean(dim=1)  # [batch, n_osc]
            x = self.projections[i](x)  # [batch, next_size]
            x = x.unsqueeze(1).expand(-1, t.shape[1], -1)  # [batch, seq, next_size]
            
            # Couche d'oscillateurs suivante (modulée par x)
            osc_out = self.layers[i+1](t)  # [batch, seq, n_osc]
            x = x * osc_out  # Modulation
        
        # Sortie finale
        x = x.mean(dim=1)  # [batch, n_osc]
        out = self.output_layer(x)  # [batch, 1]
        
        return out.squeeze(-1)


# ══════════════════════════════════════════════════════════════════
# MLP DE RÉFÉRENCE
# ══════════════════════════════════════════════════════════════════

class MLP(nn.Module):
    def __init__(self, in_dim=1, hidden=[128, 128], act='relu'):
        super().__init__()
        acts = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),
        }
        activation = acts.get(act, nn.ReLU())
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), activation]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


# ══════════════════════════════════════════════════════════════════
# JEUX DE DONNÉES (8 TÂCHES)
# ══════════════════════════════════════════════════════════════════

class Tasks:
    @staticmethod
    def task1_sine(n_samples=1000, sr=1000):
        duration = 1.0
        n_points = int(duration * sr)
        t = np.linspace(0, duration, n_points, dtype=np.float32)
        f = 50.0
        signal = np.sin(2*np.pi*f*t)
        X = torch.from_numpy(t.reshape(-1, 1))
        y = torch.from_numpy(signal)
        return X, y, "Sinusoïde pure (50 Hz)"

    @staticmethod
    def task2_chirp(n_samples=1000, sr=1000):
        duration = 1.0
        n_points = int(duration * sr)
        t = np.linspace(0, duration, n_points, dtype=np.float32)
        f0, f1 = 10.0, 200.0
        k = (f1 - f0) / duration
        phase = 2*np.pi*(f0*t + 0.5*k*t**2)
        signal = np.sin(phase)
        X = torch.from_numpy(t.reshape(-1, 1))
        y = torch.from_numpy(signal)
        return X, y, "Chirp (10→200 Hz)"
    
    @staticmethod
    def task3_am_signal(n_samples=1000, sr=1000):
        """
        Tâche 3 : Signal avec modulation d'amplitude
        """
        duration = 1.0
        n_points = int(duration * sr)
        t = np.linspace(0, duration, n_points, dtype=np.float32)
        
        # Porteuse 100 Hz, modulation 5 Hz
        carrier = 100.0
        mod_freq = 5.0
        signal = (1 + 0.5*np.sin(2*np.pi*mod_freq*t)) * np.sin(2*np.pi*carrier*t)
        
        X = torch.from_numpy(t.reshape(-1, 1))
        y = torch.from_numpy(signal)
        
        return X, y, "AM (100 Hz ± 5 Hz)"
    
    @staticmethod
    def task4_composite_signal(n_samples=1000, sr=1000):
        """
        Tâche 4 : Signal composite (plusieurs fréquences)
        """
        duration = 1.0
        n_points = int(duration * sr)
        t = np.linspace(0, duration, n_points, dtype=np.float32)
        freqs = [20.0, 60.0, 150.0]
        signal = sum(np.sin(2*np.pi*f*t) for f in freqs) / len(freqs)
        X = torch.from_numpy(t.reshape(-1, 1))
        y = torch.from_numpy(signal.astype(np.float32))
        return X, y, f"Signal composite {freqs} Hz"

    @staticmethod
    def task5_square_like(n_samples=1000, sr=1000):
        duration = 1.0
        n_points = int(duration * sr)
        t = np.linspace(0, duration, n_points, dtype=np.float32)
        # Onde carrée approchée par somme de sinus impairs
        signal = np.zeros_like(t)
        for k in range(1, 10, 2):
            signal += (1.0/k) * np.sin(2*np.pi*k*10*t)
        signal = (4/np.pi) * signal
        X = torch.from_numpy(t.reshape(-1, 1))
        y = torch.from_numpy(signal.astype(np.float32))
        return X, y, "Onde carrée (approx.)"

    @staticmethod
    def task6_saw_like(n_samples=1000, sr=1000):
        duration = 1.0
        n_points = int(duration * sr)
        t = np.linspace(0, duration, n_points, dtype=np.float32)
        # Onde en dents de scie (approx Fourier)
        signal = np.zeros_like(t)
        for k in range(1, 10):
            signal += ((-1)**(k+1)) * (1.0/k) * np.sin(2*np.pi*k*10*t)
        signal = (2/np.pi) * signal
        X = torch.from_numpy(t.reshape(-1, 1))
        y = torch.from_numpy(signal.astype(np.float32))
        return X, y, "Onde en dents de scie (approx.)"

    @staticmethod
    def task7_nonlinear(n_samples=1000, sr=1000):
        duration = 1.0
        n_points = int(duration * sr)
        t = np.linspace(0, duration, n_points, dtype=np.float32)
        x = 2 * (t - 0.5)  # x ∈ [-1,1]
        signal = np.tanh(3*x) + 0.5*(x**2)
        X = torch.from_numpy(x.reshape(-1, 1))
        y = torch.from_numpy(signal.astype(np.float32))
        return X, y, "Non-linéaire : tanh(3x) + 0.5x²"
    
    @staticmethod
    def task8_noisy_sine(n_samples=1000, sr=1000, snr_db=10):
        """
        Tâche 8 : Sinusoïde bruitée (test de robustesse)
        """
        duration = 1.0
        n_points = int(duration * sr)
        t = np.linspace(0, duration, n_points, dtype=np.float32)
        
        # Signal
        f = 40.0
        signal_clean = np.sin(2*np.pi*f*t)
        
        # Ajouter bruit
        signal_power = np.mean(signal_clean**2)
        noise_power = signal_power / (10**(snr_db/10))
        noise = np.random.randn(n_points) * np.sqrt(noise_power)
        signal = signal_clean + noise
        
        X = torch.from_numpy(t.reshape(-1, 1))
        y = torch.from_numpy(signal.astype(np.float32))
        
        return X, y, f"Sinusoïde bruitée (40 Hz, SNR={snr_db}dB)"


# ══════════════════════════════════════════════════════════════════
# ENTRAÎNEMENT ET ÉVALUATION
# ══════════════════════════════════════════════════════════════════

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model(model, X, y, epochs=500, lr=1e-3, batch_size=32, verbose=False):
    """Entraîne un modèle sur les données"""
    model = model.to(device)
    model.train()
    
    X, y = X.to(device), y.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    
    n_samples = len(X)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    best_loss = float('inf')
    best_state = None
    
    pbar = trange(epochs, desc="Training", leave=False)
    start_time = time.time()
    
    for epoch in pbar:
        # Shuffle
        indices = torch.randperm(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        epoch_loss = 0.0
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i+1) * batch_size, n_samples)
            
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]
            
            optimizer.zero_grad()
            
            # Forward
            if isinstance(model, DeepOscillatorNetwork):
                # Pour oscillateurs : entrée = temps (garder [batch, 1])
                pred = model(X_batch)
            else:
                # Pour MLP : entrée = features
                pred = model(X_batch)
            
            loss = loss_fn(pred, y_batch)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * (end_idx - start_idx)
        
        epoch_loss /= n_samples
        pbar.set_postfix(loss=f"{epoch_loss:.6f}")
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    
    train_time = time.time() - start_time
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return best_loss, train_time


def evaluate_model(model, X, y):
    """Évalue un modèle"""
    model.eval()
    X, y = X.to(device), y.to(device)
    
    with torch.no_grad():
        if isinstance(model, DeepOscillatorNetwork):
            pred = model(X)
        else:
            pred = model(X)
        
        mse = nn.MSELoss()(pred, y).item()
        mae = nn.L1Loss()(pred, y).item()
        
        # SNR
        signal_power = torch.mean(y**2).item()
        noise_power = torch.mean((y - pred)**2).item()
        snr_db = 10 * np.log10(signal_power / (noise_power + 1e-12))
        
        # Correlation
        pred_np = pred.cpu().numpy()
        y_np = y.cpu().numpy()
        correlation = np.corrcoef(pred_np, y_np)[0, 1]
    
    return {
        'mse': mse,
        'mae': mae,
        'snr_db': snr_db,
        'correlation': correlation,
        'predictions': pred.cpu().numpy(),
        'targets': y.cpu().numpy()
    }


# ══════════════════════════════════════════════════════════════════
# COMPARAISON COMPLÈTE
# ══════════════════════════════════════════════════════════════════

def run_comparison():
    print("="*80)
    print("COMPARAISON : RÉSEAUX D'OSCILLATEURS vs PERCEPTRONS CLASSIQUES")
    print("="*80)
    print()

    tasks = [
        Tasks.task1_sine,
        Tasks.task2_chirp,
        Tasks.task3_am_signal,
        Tasks.task4_composite_signal,
        Tasks.task5_square_like,
        Tasks.task6_saw_like,
        Tasks.task7_nonlinear,
        Tasks.task8_noisy_sine,
    ]

    results = []

    for task_id, task_fn in enumerate(tasks, start=1):
        print("="*80)
        print(f"TÂCHE {task_id}/8")
        print("="*80)
        
        X, y, task_name = task_fn()
        print(f"📋 {task_name}")
        print(f"   Données : {len(X)} échantillons\n")
        
        # Modèle oscillateurs
        osc = DeepOscillatorNetwork([
            (32, 5, 200),
            (48, 20, 800),
            (32, 50, 1200),
        ], learnable_freq=False)
        n_params_osc = count_params(osc)
        print("🔄 Réseau d'Oscillateurs Laplaciens")
        print(f"   Paramètres : {n_params_osc}\n")
        
        loss_osc, time_osc = train_model(osc, X, y, epochs=500, lr=5e-3, verbose=True)
        eval_osc = evaluate_model(osc, X, y)
        
        # Modèle MLP
        mlp = MLP(in_dim=1, hidden=[128, 128], act='tanh')
        n_params_mlp = count_params(mlp)
        print("🧠 MLP classique (tanh)")
        print(f"   Paramètres : {n_params_mlp}\n")
        
        loss_mlp, time_mlp = train_model(mlp, X, y, epochs=500, lr=1e-3, verbose=True)
        eval_mlp = evaluate_model(mlp, X, y)
        
        # Résultats
        print(f"SNR Oscillateurs : {eval_osc['snr_db']:.2f} dB | MSE={eval_osc['mse']:.6f}")
        print(f"SNR MLP          : {eval_mlp['snr_db']:.2f} dB | MSE={eval_mlp['mse']:.6f}")
        winner = "Oscillateurs" if eval_osc['snr_db'] > eval_mlp['snr_db'] else "MLP"
        delta_snr = eval_osc['snr_db'] - eval_mlp['snr_db']
        print(f"➡️  Gagnant : {winner} (ΔSNR = {delta_snr:+.2f} dB)")
        print()
        
        results.append({
            'task': task_name,
            'osc_snr': eval_osc['snr_db'],
            'mlp_snr': eval_mlp['snr_db'],
            'osc_time': time_osc,
            'mlp_time': time_mlp,
            'osc_params': n_params_osc,
            'mlp_params': n_params_mlp,
            'winner': winner,
            'delta_snr': delta_snr
        })
    
    # Résumé final
    print("\n" + "="*80)
    print("RÉSUMÉ GLOBAL")
    print("="*80)
    print()
    print(f"{'Tâche':<40} {'Osc SNR':>10} {'MLP SNR':>10} {'Δ SNR':>10} {'Gagnant':>12}")
    for r in results:
        print(f"{r['task']:<40} {r['osc_snr']:>10.2f} {r['mlp_snr']:>10.2f} {r['delta_snr']:>10.2f} {r['winner']:>12}")
    print()
    return results


if __name__ == "__main__":
    results = run_comparison()
