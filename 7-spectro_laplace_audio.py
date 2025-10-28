#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Laplace Audio - MODÈLE HYBRIDE
--------------------------------
Solution finale : Oscillateurs harmoniques + Composante de bruit

Signal = Partie déterministe (oscillateurs) + Partie stochastique (bruit)

Usage:
    python laplace_hybrid.py --wav audio.wav --seconds 3.0
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import argparse
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from scipy.io import wavfile
from scipy.signal import resample, spectrogram
from scipy.io.wavfile import write as wav_write


class HybridLaplaceModel(nn.Module):
    """
    Modèle hybride:
    y(t) = y_deterministe(t) + y_stochastique(t)
    
    - y_deterministe : Oscillateurs laplaciens (harmoniques)
    - y_stochastique : Réseau apprend à prédire le résidu
    """
    def __init__(self, n_units=384, sr=8000, max_s=0.3, duration=10.0, n_samples=24000,
                 net_channels=64, net_layers=3, net_dilations=None, net_kernel=5):
        """
        Args:
            n_units: Nombre d'oscillateurs
            sr: Sample rate
            max_s: Damping maximum
            duration: Durée du signal
            n_samples: Nombre de samples
            net_channels: Nombre de canaux dans le réseau (default: 64)
            net_layers: Nombre de couches dilated (default: 3)
            net_dilations: Liste des dilations (default: [2, 4, 8, 16, ...])
            net_kernel: Taille des kernels dilated (default: 5)
        """
        super().__init__()
        self.n_units = n_units
        self.max_s = max_s
        self.duration = duration
        self.sr = sr
        self.n_samples = n_samples
        self.net_channels = net_channels
        self.net_layers = net_layers
        
        # Partie oscillateurs
        nyquist = sr / 2.0
        f_min = 20.0
        f_max = 0.49 * nyquist
        
        linear = np.linspace(0, 1, n_units)
        warped = linear ** 0.8
        freqs = f_min + (f_max - f_min) * warped
        freqs = torch.from_numpy(freqs).float()
        
        w = 2 * math.pi * freqs
        
        self.register_buffer("omega", w)
        self.register_buffer("freqs_hz", freqs)
        
        self.a = nn.Parameter(torch.zeros(n_units))
        self.b = nn.Parameter(torch.zeros(n_units))
        
        freq_norm = (freqs - f_min) / (f_max - f_min)
        s_init = 0.005 * (1 - freq_norm)**2 + 1e-5
        self.log_s = nn.Parameter(torch.log(s_init + 1e-9))
        
        self.dc = nn.Parameter(torch.tensor(0.0))
        self.tau = nn.Parameter(torch.tensor(0.0))
        
        # Partie stochastique : réseau convolutif CONFIGURABLE
        # Générer les dilations si non spécifiées
        # FIX: Commencer à 2^1 au lieu de 2^0 pour éviter dilation=1
        if net_dilations is None:
            net_dilations = [2**(i+1) for i in range(net_layers)]
        else:
            net_dilations = net_dilations[:net_layers]
        
        self.net_dilations = net_dilations
        
        # Construction dynamique du réseau
        layers = []
        
        # Embedding initial
        layers.append(nn.Conv1d(1, net_channels, kernel_size=15, padding=7))
        layers.append(nn.Tanh())
        
        # Couches dilated
        for dilation in net_dilations:
            padding = (net_kernel - 1) * dilation // 2
            layers.append(nn.Conv1d(net_channels, net_channels, 
                                   kernel_size=net_kernel, 
                                   padding=padding, 
                                   dilation=dilation))
            layers.append(nn.Tanh())
        
        # Projection vers signal
        layers.append(nn.Conv1d(net_channels, net_channels // 2, kernel_size=5, padding=2))
        layers.append(nn.Tanh())
        layers.append(nn.Conv1d(net_channels // 2, 1, kernel_size=3, padding=1))
        
        self.noise_net = nn.Sequential(*layers)
        
        # Weight pour mélange (apprenable)
        self.mix_weight = nn.Parameter(torch.tensor(0.5))
    
    def get_damping(self):
        s = F.softplus(self.log_s)
        return torch.clamp(s, min=1e-9, max=self.max_s)
    
    def forward_deterministic(self, t):
        """Partie déterministe (oscillateurs)"""
        t_shifted = t + self.tau
        s = self.get_damping()
        
        decay = torch.exp(-s.view(1, 1, -1) * t_shifted.unsqueeze(2))
        phase = self.omega.view(1, 1, -1) * t_shifted.unsqueeze(2)
        
        sin_basis = decay * torch.sin(phase)
        cos_basis = decay * torch.cos(phase)
        
        return (self.a * sin_basis + self.b * cos_basis).sum(dim=2) + self.dc
    
    def forward_stochastic(self, t, deterministic_part):
        """
        Partie stochastique (réseau).
        Prend en entrée la partie déterministe pour prédire le résidu.
        """
        # Reshape pour conv1d : [B, C, T]
        x = deterministic_part.unsqueeze(1)  # [B, 1, T]
        
        # Réseau prédit le résidu
        residual = self.noise_net(x).squeeze(1)  # [B, T]
        
        return residual
    
    def forward(self, t):
        # Partie déterministe
        y_det = self.forward_deterministic(t)
        
        # Partie stochastique
        y_sto = self.forward_stochastic(t, y_det)
        
        # Mélange avec poids apprenable (sigmoid pour [0, 1])
        alpha = torch.sigmoid(self.mix_weight)
        
        return alpha * y_det + (1 - alpha) * y_sto
    
    def init_from_fft(self, signal_np):
        """Init partie déterministe"""
        T = len(signal_np)
        fft = np.fft.rfft(signal_np)
        
        self.dc.data.fill_(float(fft[0].real / T))
        
        freqs_fft = np.fft.rfftfreq(T, d=1.0/self.sr)
        freqs_model = self.freqs_hz.cpu().numpy()
        
        for k in range(self.n_units):
            f_target = freqs_model[k]
            idx = np.argmin(np.abs(freqs_fft - f_target))
            
            if idx < len(fft):
                self.a.data[k] = float(-2 * fft[idx].imag / T)
                self.b.data[k] = float(2 * fft[idx].real / T)
        
        n_params = sum(p.numel() for p in self.noise_net.parameters())
        print(f"   {self.n_units} oscillateurs + réseau stochastique")
        print(f"   Réseau : {self.net_layers} couches, {self.net_channels} canaux, dilations={self.net_dilations}")
        print(f"   Params réseau : {n_params:,}")


def load_audio_mono(path, target_sr=8000):
    sr, data = wavfile.read(path)
    
    if data.dtype == np.int16:
        x = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        x = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        x = (data.astype(np.float32) - 128.0) / 128.0
    else:
        x = data.astype(np.float32)
    
    if x.ndim == 2:
        x = x.mean(axis=1)
    
    if sr != target_sr:
        n_new = int(len(x) * (target_sr / sr))
        x = resample(x, n_new)
    
    x = x - np.mean(x)
    x = x / (np.max(np.abs(x)) + 1e-9)
    
    return target_sr, x


def train_model(model, t, signal, epochs, lr, s_reg, target_decay):
    # Optimizers séparés pour oscillateurs vs réseau
    osc_params = [model.a, model.b, model.dc, model.tau, model.log_s]
    net_params = list(model.noise_net.parameters()) + [model.mix_weight]
    
    optimizer_osc = optim.AdamW(osc_params, lr=lr, weight_decay=1e-6)
    optimizer_net = optim.AdamW(net_params, lr=lr*2, weight_decay=1e-5)  # LR plus élevé pour réseau
    
    def lr_lambda(epoch):
        warmup = int(0.1 * epochs)
        if epoch < warmup:
            return (epoch + 1) / warmup
        prog = (epoch - warmup) / (epochs - warmup)
        return 0.5 * (1 + np.cos(np.pi * prog))
    
    scheduler_osc = optim.lr_scheduler.LambdaLR(optimizer_osc, lr_lambda)
    scheduler_net = optim.lr_scheduler.LambdaLR(optimizer_net, lr_lambda)
    
    losses = []
    best_loss = float('inf')
    best_state = None
    
    print(f"\n   Training HYBRIDE (oscillateurs + réseau stochastique)...")
    
    for ep in trange(epochs, desc="Train"):
        optimizer_osc.zero_grad()
        optimizer_net.zero_grad()
        
        pred = model(t)
        
        # MSE principale
        loss = F.mse_loss(pred, signal)
        
        # Régularisation légère sur oscillateurs
        s_vals = model.get_damping()
        loss += s_reg * 0.5 * s_vals.pow(2).mean()
        
        # Penalty HF
        hf_mask = model.freqs_hz > 1000
        if hf_mask.sum() > 0:
            loss += s_reg * 2.0 * s_vals[hf_mask].pow(2).mean()
        
        # Adaptive decay
        with torch.no_grad():
            mean_decay = torch.exp(-s_vals * model.duration).mean()
        
        if mean_decay < target_decay:
            penalty = (target_decay - mean_decay) / target_decay
            loss += s_reg * 5.0 * penalty * s_vals.pow(2).mean()
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(osc_params, 1.0)
        torch.nn.utils.clip_grad_norm_(net_params, 1.0)
        
        optimizer_osc.step()
        optimizer_net.step()
        
        scheduler_osc.step()
        scheduler_net.step()
        
        if ep == int(0.2 * epochs):
            model.tau.requires_grad_(False)
        
        loss_val = loss.item()
        losses.append(loss_val)
        
        if loss_val < best_loss:
            best_loss = loss_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if not np.isfinite(loss_val) or loss_val > 1e3:
            if best_state:
                model.load_state_dict(best_state)
            break
        
        if ep % 500 == 0 and ep > 0:
            with torch.no_grad():
                mae = torch.abs(pred - signal).mean().item()
                alpha = torch.sigmoid(model.mix_weight).item()
            print(f"\n   Epoch {ep}: Loss={loss_val:.6f}, MAE={mae:.4f}, alpha={alpha:.3f}")
    
    if best_state:
        model.load_state_dict(best_state)
        print(f"\n   ✓ Best: {best_loss:.6f}")
    
    return losses


def plot_results(t_np, signal_np, pred_np, losses, sr, model):
    fig = plt.figure(figsize=(18, 14))
    
    # Décomposition
    with torch.no_grad():
        t_t = torch.from_numpy(t_np).unsqueeze(0)
        y_det = model.forward_deterministic(t_t).numpy().squeeze()
        alpha = torch.sigmoid(model.mix_weight).item()
    
    y_sto = pred_np - alpha * y_det
    
    # Waveforms
    ax1 = plt.subplot(4, 3, 1)
    ax1.plot(t_np, signal_np, 'gray', lw=0.8, alpha=0.6, label='Target')
    ax1.plot(t_np, pred_np, 'red', lw=0.8, alpha=0.9, label='Hybrid')
    ax1.set_title('Waveform Total', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    ax2 = plt.subplot(4, 3, 2)
    ax2.plot(t_np, y_det, 'blue', lw=0.8, alpha=0.7, label='Déterministe')
    ax2.set_title(f'Partie Oscillateurs (α={alpha:.2f})', fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    ax3 = plt.subplot(4, 3, 3)
    ax3.plot(t_np, y_sto, 'green', lw=0.8, alpha=0.7, label='Stochastique')
    ax3.set_title(f'Partie Réseau (1-α={1-alpha:.2f})', fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # Loss
    ax4 = plt.subplot(4, 3, 4)
    ax4.plot(losses, lw=1.5)
    ax4.set_yscale('log')
    ax4.set_title('Loss', fontweight='bold')
    ax4.grid(alpha=0.3)
    
    # Error
    error = np.abs(pred_np - signal_np)
    ax5 = plt.subplot(4, 3, 5)
    ax5.plot(t_np, error, 'blue', lw=0.5)
    ax5.set_title(f'Error (MAE={error.mean():.4f})', fontweight='bold')
    ax5.grid(alpha=0.3)
    
    # Spectra
    fft_s = np.abs(np.fft.rfft(signal_np))
    fft_p = np.abs(np.fft.rfft(pred_np))
    fft_d = np.abs(np.fft.rfft(y_det))
    fft_n = np.abs(np.fft.rfft(y_sto))
    freqs = np.fft.rfftfreq(len(signal_np), 1.0/sr)
    
    ax6 = plt.subplot(4, 3, 6)
    ax6.semilogx(freqs + 1, fft_s / (fft_s.max() + 1e-9), 'gray', lw=1.2, alpha=0.7, label='Target')
    ax6.semilogx(freqs + 1, fft_p / (fft_p.max() + 1e-9), 'red', lw=1.2, alpha=0.9, label='Hybrid')
    ax6.semilogx(freqs + 1, fft_d / (fft_d.max() + 1e-9), 'blue', lw=1, alpha=0.6, label='Det')
    ax6.semilogx(freqs + 1, fft_n / (fft_n.max() + 1e-9), 'green', lw=1, alpha=0.6, label='Sto')
    ax6.set_xlim(100, sr/2)
    ax6.set_title('Spectra', fontweight='bold')
    ax6.legend(fontsize=8)
    ax6.grid(alpha=0.3)
    
    # Spectrograms
    f_s, t_s, Sxx_s = spectrogram(signal_np, sr, nperseg=256, noverlap=224)
    ax7 = plt.subplot(4, 3, 7)
    ax7.pcolormesh(t_s, f_s, 10*np.log10(Sxx_s + 1e-10), shading='gouraud', cmap='viridis')
    ax7.set_ylim(0, sr/2)
    ax7.set_title('Spectrogram (Target)', fontweight='bold')
    
    f_p, t_p, Sxx_p = spectrogram(pred_np, sr, nperseg=256, noverlap=224)
    ax8 = plt.subplot(4, 3, 8)
    ax8.pcolormesh(t_p, f_p, 10*np.log10(Sxx_p + 1e-10), shading='gouraud', cmap='viridis')
    ax8.set_ylim(0, sr/2)
    ax8.set_title('Spectrogram (Hybrid)', fontweight='bold')
    
    f_d, t_d, Sxx_d = spectrogram(y_det, sr, nperseg=256, noverlap=224)
    ax9 = plt.subplot(4, 3, 9)
    ax9.pcolormesh(t_d, f_d, 10*np.log10(Sxx_d + 1e-10), shading='gouraud', cmap='viridis')
    ax9.set_ylim(0, sr/2)
    ax9.set_title('Spectrogram (Déterministe)', fontweight='bold')
    
    f_n, t_n, Sxx_n = spectrogram(y_sto, sr, nperseg=256, noverlap=224)
    ax10 = plt.subplot(4, 3, 10)
    ax10.pcolormesh(t_n, f_n, 10*np.log10(np.abs(Sxx_n) + 1e-10), shading='gouraud', cmap='viridis')
    ax10.set_ylim(0, sr/2)
    ax10.set_title('Spectrogram (Stochastique)', fontweight='bold')
    
    # Analysis
    with torch.no_grad():
        s_vals = model.get_damping().cpu().numpy()
        freqs_m = model.freqs_hz.cpu().numpy()
        a = model.a.cpu().numpy()
        b = model.b.cpu().numpy()
        amp = np.sqrt(a**2 + b**2)
    
    ax11 = plt.subplot(4, 3, 11)
    ax11.scatter(freqs_m, amp, s=15, alpha=0.6, c=amp, cmap='viridis')
    ax11.set_xscale('log')
    ax11.set_yscale('log')
    ax11.set_title('Oscillateurs Amplitudes', fontweight='bold')
    ax11.grid(alpha=0.3)
    
    ax12 = plt.subplot(4, 3, 12)
    zoom = int(0.05 * sr)
    ax12.plot(t_np[:zoom], signal_np[:zoom], 'gray', lw=1.5, alpha=0.7, label='Target')
    ax12.plot(t_np[:zoom], pred_np[:zoom], 'red', lw=1.5, label='Hybrid')
    ax12.set_title('Zoom 50ms', fontweight='bold')
    ax12.legend()
    ax12.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Laplace Audio Hybrid - Oscillateurs + Réseau Stochastique',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  
  # Configuration par défaut (3 couches, 64 canaux, dilations=[2,4,8])
  python laplace_hybrid.py --wav test.wav --seconds 3.0
  
  # Réseau profond (6 couches, dilations=[2,4,8,16,32,64])
  python laplace_hybrid.py --wav test.wav --seconds 3.0 --net-layers 6
  
  # Réseau large (128 canaux)
  python laplace_hybrid.py --wav test.wav --seconds 3.0 --net-channels 128
  
  # Très profond avec dilations personnalisées
  python laplace_hybrid.py --wav test.wav --seconds 3.0 --net-layers 8 --net-dilations 1,2,4,8,16,32,64,128
  
  # Léger pour CPU (2 couches, 32 canaux)
  python laplace_hybrid.py --wav test.wav --seconds 3.0 --net-layers 2 --net-channels 32
        """
    )
    
    # Audio args
    parser.add_argument("--wav", required=True, help="Fichier audio d'entrée")
    parser.add_argument("--sr", type=int, default=8000, help="Sample rate (default: 8000)")
    parser.add_argument("--seconds", type=float, default=3.0, help="Durée en secondes (default: 3.0)")
    
    # Oscillateurs args
    parser.add_argument("--units", type=int, default=384, help="Nombre d'oscillateurs (default: 384)")
    parser.add_argument("--max-s", type=float, default=0.3, help="Damping maximum (default: 0.3)")
    parser.add_argument("--s-reg", type=float, default=0.005, help="Régularisation damping (default: 0.005)")
    parser.add_argument("--target-decay", type=float, default=0.7, help="Decay target (default: 0.7)")
    
    # Network args
    parser.add_argument("--net-channels", type=int, default=64, 
                       help="Nombre de canaux du réseau (default: 64, plus=meilleur mais plus lent)")
    parser.add_argument("--net-layers", type=int, default=3,
                       help="Nombre de couches dilated (default: 3, plus=champ réceptif plus large)")
    parser.add_argument("--net-dilations", type=str, default=None,
                       help="Dilations personnalisées, séparées par virgules (ex: '2,4,8,16')")
    parser.add_argument("--net-kernel", type=int, default=5,
                       help="Taille des kernels dilated (default: 5)")
    
    # Training args
    parser.add_argument("--epochs", type=int, default=3000, help="Nombre d'epochs (default: 3000)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    
    args = parser.parse_args()
    
    # Parse dilations si spécifiées
    net_dilations = None
    if args.net_dilations:
        try:
            net_dilations = [int(d.strip()) for d in args.net_dilations.split(',')]
            print(f"Dilations personnalisées : {net_dilations}")
        except:
            print(f"⚠️  Format invalide pour --net-dilations, utilisation des valeurs par défaut")
    
    print("="*70)
    print("LAPLACE HYBRIDE (Oscillateurs + Réseau Stochastique)")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    
    sr, x = load_audio_mono(args.wav, args.sr)
    n = int(args.seconds * sr)
    x = x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))
    
    duration = len(x) / sr
    t = np.linspace(0, duration, len(x), endpoint=False, dtype=np.float32)
    t_t = torch.from_numpy(t).unsqueeze(0).to(device)
    signal = torch.from_numpy(x).unsqueeze(0).to(device)
    
    model = HybridLaplaceModel(
        n_units=args.units, 
        sr=sr, 
        max_s=args.max_s, 
        duration=duration, 
        n_samples=len(x),
        net_channels=args.net_channels,
        net_layers=args.net_layers,
        net_dilations=net_dilations,
        net_kernel=args.net_kernel
    ).to(device)
    
    model.init_from_fft(x)
    
    losses = train_model(model, t_t, signal, args.epochs, args.lr, args.s_reg, args.target_decay)
    
    with torch.no_grad():
        pred = model(t_t)
        mse = F.mse_loss(pred, signal).item()
        mae = torch.abs(pred - signal).mean().item()
        alpha = torch.sigmoid(model.mix_weight).item()
    
    print(f"\n✓ Final: MSE={mse:.6e}, MAE={mae:.6f}, mix={alpha:.3f}")
    
    pred_np = pred.cpu().numpy().squeeze()
    signal_np = signal.cpu().numpy().squeeze()
    
    # Nom de fichier avec config
    fname = f"laplace_HYBRID_L{args.net_layers}_C{args.net_channels}.wav"
    wav_write(fname, sr, (np.clip(pred_np, -1, 1) * 32767).astype(np.int16))
    print(f"   → {fname}")
    
    fig = plot_results(t, signal_np, pred_np, losses, sr, model)
    
    pname = f"laplace_HYBRID_L{args.net_layers}_C{args.net_channels}.png"
    fig.savefig(pname, dpi=150)
    print(f"   → {pname}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()