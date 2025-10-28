#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
======================================================================
RÉSEAUX D'OSCILLATEURS vs PERCEPTRONS CLASSIQUES (VERSION MULTI-BANDES)
======================================================================

NOUVELLE ARCHITECTURE (conforme à la demande) :
  1) CNNFilterBank : séparation en bandes + débruitage (Conv1d)
  2) LaplacianOscillators : UNE SEULE couche d'oscillateurs, découpée en sous-banques (une par bande)
  3) Recomposer : recomposition (Conv1d 1x1)

Comparaison conservée avec un MLP classique.

Tâches : identiques à la version précédente (8 tâches synthétiques).
"""

import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import trange
from typing import List, Tuple, Dict

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cpu")

# ══════════════════════════════════════════════════════════════════
# MLP DE RÉFÉRENCE (inchangé)
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
# NOUVELLE ARCHITECTURE MULTI-BANDES
#   1) CNN de séparation + débruitage
#   2) Une seule couche d'oscillateurs (par bande)
#   3) Recomposition 1x1
# ══════════════════════════════════════════════════════════════════

class CNNFilterBank(nn.Module):
    def __init__(self, in_ch=1, bands=3):
        super().__init__()
        # Trois branches (Low/Mid/High) ; tu peux monter à 4+ bandes si besoin
        self.low  = nn.Sequential(
            nn.Conv1d(in_ch, 1, kernel_size=129, padding=64, bias=False),
            nn.BatchNorm1d(1), nn.SiLU(),
            nn.Conv1d(1, 1, kernel_size=15, padding=7, bias=False),
        )
        self.mid  = nn.Sequential(
            nn.Conv1d(in_ch, 1, kernel_size=63,  padding=31, bias=False),
            nn.BatchNorm1d(1), nn.SiLU(),
            nn.Conv1d(1, 1, kernel_size=11, padding=5, bias=False),
        )
        self.high = nn.Sequential(
            nn.Conv1d(in_ch, 1, kernel_size=15,  padding=7, bias=False),
            nn.BatchNorm1d(1), nn.SiLU(),
            nn.Conv1d(1, 1, kernel_size=7,  padding=3, bias=False),
        )
        # Débruitage léger depthwise + pointwise
        self.denoise = nn.Sequential(
            nn.Conv1d(bands, bands, kernel_size=5, padding=2, groups=bands, bias=False),
            nn.BatchNorm1d(bands), nn.SiLU(),
            nn.Conv1d(bands, bands, kernel_size=1, bias=False)
        )

    def forward(self, x):                # x: [B,1,T]
        low  = self.low(x)               # [B,1,T]
        mid  = self.mid(x)               # [B,1,T]
        high = self.high(x)              # [B,1,T]
        z = torch.cat([low, mid, high], dim=1)  # [B,3,T]
        z = self.denoise(z)              # [B,3,T]
        return z


class LaplacianOscillators(nn.Module):
    """
    Une seule couche d’oscillateurs amortis, organisée en sous-banques par bande.
    y_k(t) = exp(-s_k t) * sin(w_k t + phi_k) * g_k(t)
    - (w_k, s_k, phi_k, gain_k) appris
    - gate g_k(t) dépend du contenu de la bande via conv 1x1
    """
    def __init__(self, bands=3, K_per_band=32, fs=8000.0,
                 w_min=2*math.pi*20, w_max=2*math.pi*3000):
        super().__init__()
        self.bands = bands
        self.K = K_per_band
        self.fs = fs
        self.w_min = w_min
        self.w_max = w_max

        # Paramètres par bande/oscillateur
        self.w_raw = nn.Parameter(torch.randn(bands, K_per_band))
        self.s_raw = nn.Parameter(torch.randn(bands, K_per_band) * 0.1)
        self.phase = nn.Parameter(torch.zeros(bands, K_per_band))
        self.gain  = nn.Parameter(torch.ones(bands, K_per_band) * 0.1)

        # Gate dépendant du signal de la bande
        self.gate = nn.Sequential(
            nn.Conv1d(1, K_per_band, kernel_size=1),
            nn.Sigmoid()
        )

    def _softplus(self, x):
        return F.softplus(x)  # s >= 0

    def _w_bounded(self, w_raw):
        return self.w_min + (self.w_max - self.w_min) * torch.sigmoid(w_raw)

    def forward(self, bands_sig, t):
        """
        bands_sig: [B, C(=bands), T]
        t:         [T] (secondes)
        return:    [B, C, T]
        """
        B, C, T = bands_sig.shape
        assert C == self.bands
        device = bands_sig.device

        t = t.to(device).view(1, 1, T)  # [1,1,T]
        out_bands = []

        for b in range(self.bands):
            x_b = bands_sig[:, b:b+1, :]              # [B,1,T]

            w_b = self._w_bounded(self.w_raw[b])      # [K]
            s_b = self._softplus(self.s_raw[b])       # [K]
            phi = self.phase[b]                       # [K]
            g   = self.gain[b]                        # [K]

            gate_bt = self.gate(x_b)                  # [B,K,T] in [0,1]

            w_bt   = w_b.view(1, self.K, 1)
            s_bt   = s_b.view(1, self.K, 1)
            phi_bt = phi.view(1, self.K, 1)
            g_bt   = g.view(1, self.K, 1)

            osc = torch.exp(-s_bt * t) * torch.sin(w_bt * t + phi_bt)  # [1,K,T]
            osc = g_bt * osc                                           # [1,K,T]

            y_b = (gate_bt * osc).sum(dim=1, keepdim=True)             # [B,1,T]
            out_bands.append(y_b)

        return torch.cat(out_bands, dim=1)  # [B,C,T]


class Recomposer(nn.Module):
    def __init__(self, bands=3):
        super().__init__()
        self.mix = nn.Conv1d(bands, 1, kernel_size=1, bias=True)

    def forward(self, y_bands):          # [B,C,T]
        return self.mix(y_bands)         # [B,1,T]


class MultiBandLaplaceNet(nn.Module):
    def __init__(self, fs=8000.0, bands=3, K_per_band=32):
        super().__init__()
        self.fs = fs
        self.frontend = CNNFilterBank(in_ch=1, bands=bands)
        self.osc      = LaplacianOscillators(bands=bands, K_per_band=K_per_band, fs=fs)
        self.recomp   = Recomposer(bands=bands)

    @torch.no_grad()
    def make_time(self, T, device):
        return torch.arange(T, device=device) / self.fs  # [T]

    def forward(self, x):
        """
        x: [B,1,T]
        return: [B,1,T]
        """
        B, C, T = x.shape
        z  = self.frontend(x)            # [B,3,T]
        t  = self.make_time(T, x.device) # [T]
        yb = self.osc(z, t)              # [B,3,T]
        y  = self.recomp(yb)             # [B,1,T]
        return y


# ══════════════════════════════════════════════════════════════════
# JEUX DE DONNÉES (8 TÂCHES, inchangés)
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
        duration = 1.0
        n_points = int(duration * sr)
        t = np.linspace(0, duration, n_points, dtype=np.float32)
        carrier = 100.0
        mod_freq = 5.0
        signal = (1 + 0.5*np.sin(2*np.pi*mod_freq*t)) * np.sin(2*np.pi*carrier*t)
        X = torch.from_numpy(t.reshape(-1, 1))
        y = torch.from_numpy(signal)
        return X, y, "AM (100 Hz ± 5 Hz)"
    
    @staticmethod
    def task4_composite_signal(n_samples=1000, sr=1000):
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
        duration = 1.0
        n_points = int(duration * sr)
        t = np.linspace(0, duration, n_points, dtype=np.float32)
        f = 40.0
        signal_clean = np.sin(2*np.pi*f*t)
        signal_power = np.mean(signal_clean**2)
        noise_power = signal_power / (10**(snr_db/10))
        noise = np.random.randn(n_points) * np.sqrt(noise_power)
        signal = signal_clean + noise
        X = torch.from_numpy(t.reshape(-1, 1))
        y = torch.from_numpy(signal.astype(np.float32))
        return X, y, f"Sinusoïde bruitée (40 Hz, SNR={snr_db}dB)"


# ══════════════════════════════════════════════════════════════════
# ENTRAÎNEMENT & ÉVALUATION (adaptés pour séquence complète)
# ══════════════════════════════════════════════════════════════════

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model(model, X, y, epochs=500, lr=1e-3, batch_size=32, verbose=False):
    """
    - Si modèle = MultiBandLaplaceNet : on entraîne sur la séquence entière [1,1,T]
    - Si modèle = MLP : entraînement point-par-point comme avant
    """
    model = model.to(device)
    model.train()

    X, y = X.to(device), y.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    pbar = trange(epochs, desc="Training", leave=False)
    start_time = time.time()
    best_loss = float('inf')
    best_state = None

    if isinstance(model, MultiBandLaplaceNet):
        x_seq = X.view(1, 1, -1)   # [1,1,T]
        y_seq = y.view(1, 1, -1)   # [1,1,T]
        for epoch in pbar:
            optimizer.zero_grad()
            pred = model(x_seq)            # [1,1,T]
            loss = loss_fn(pred.squeeze(0).squeeze(0), y_seq.squeeze(0).squeeze(0))
            loss.backward()
            optimizer.step()

            cur = loss.item()
            pbar.set_postfix(loss=f"{cur:.6f}")
            if cur < best_loss:
                best_loss = cur
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    else:
        # MLP : batching sur points (comme avant)
        n_samples = len(X)
        n_batches = (n_samples + batch_size - 1) // batch_size
        for epoch in pbar:
            # Shuffle
            indices = torch.randperm(n_samples)
            X_shuf = X[indices]; y_shuf = y[indices]
            epoch_loss = 0.0

            for i in range(n_batches):
                s = i * batch_size
                e = min((i+1)*batch_size, n_samples)
                xb = X_shuf[s:e]
                yb = y_shuf[s:e]
                optimizer.zero_grad()
                pred = model(xb)           # [B]
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * (e - s)

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
    model.eval()
    X, y = X.to(device), y.to(device)

    with torch.no_grad():
        if isinstance(model, MultiBandLaplaceNet):
            x_seq = X.view(1,1,-1)
            pred = model(x_seq).view(-1)      # [T]
        else:
            pred = model(X)                    # [T]

        mse = nn.MSELoss()(pred, y).item()
        mae = nn.L1Loss()(pred, y).item()

        signal_power = torch.mean(y**2).item()
        noise_power = torch.mean((y - pred)**2).item()
        snr_db = 10 * np.log10(signal_power / (noise_power + 1e-12))

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
    print("COMPARAISON : MULTI-BAND LAPLACE NET vs MLP")
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
        
        # ----- Modèle Multi-Band Laplace (NOUVEAU) -----
        osc = MultiBandLaplaceNet(fs=1000.0, bands=3, K_per_band=32)
        n_params_osc = count_params(osc)
        print("🔄 Multi-Band Laplace Net (CNN → Osc layer unique → 1x1)")
        print(f"   Paramètres : {n_params_osc}\n")
        loss_osc, time_osc = train_model(osc, X, y, epochs=500, lr=2e-3, verbose=True)
        eval_osc = evaluate_model(osc, X, y)
        
        # ----- Modèle MLP -----
        mlp = MLP(in_dim=1, hidden=[128, 128], act='tanh')
        n_params_mlp = count_params(mlp)
        print("🧠 MLP classique (tanh)")
        print(f"   Paramètres : {n_params_mlp}\n")
        loss_mlp, time_mlp = train_model(mlp, X, y, epochs=500, lr=1e-3, verbose=True)
        eval_mlp = evaluate_model(mlp, X, y)
        
        # ----- Résultats -----
        print(f"SNR Multi-Band : {eval_osc['snr_db']:.2f} dB | MSE={eval_osc['mse']:.6f}")
        print(f"SNR MLP        : {eval_mlp['snr_db']:.2f} dB | MSE={eval_mlp['mse']:.6f}")
        winner = "Multi-Band" if eval_osc['snr_db'] > eval_mlp['snr_db'] else "MLP"
        delta_snr = eval_osc['snr_db'] - eval_mlp['snr_db']
        print(f"➡️  Gagnant : {winner} (ΔSNR = {delta_snr:+.2f} dB)\n")
        
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
    print(f"{'Tâche':<40} {'MB SNR':>10} {'MLP SNR':>10} {'Δ SNR':>10} {'Gagnant':>12}")
    for r in results:
        print(f"{r['task']:<40} {r['osc_snr']:>10.2f} {r['mlp_snr']:>10.2f} {r['delta_snr']:>10.2f} {r['winner']:>12}")
    print()
    return results


if __name__ == "__main__":
    results = run_comparison()
