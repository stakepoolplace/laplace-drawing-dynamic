#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
======================================================================
LAPLACE RNN VRAIMENT RÉCURRENT - Multi-échelle
======================================================================

NOUVELLE ARCHITECTURE : Vraie récurrence avec réinjection du signal

Au lieu de :
    y(t) = exp(-s·t) · [a·sin(ω·t) + b·cos(ω·t)]  (feed-forward)

On utilise :
    h[t] = tanh(W_h · h[t-1] + W_x · x[t] + b)
    y[t] = W_o · h[t]

Avec des états cachés qui modélisent des dynamiques oscillatoires
amorties apprises plutôt que codées en dur.

AVANTAGES :
- Vraie récurrence temporelle
- Réinjection du signal (peut apprendre des corrections itératives)
- États cachés persistants
- Plus de flexibilité d'apprentissage

STRUCTURE MULTI-ÉCHELLE :
- Plusieurs RNN avec différentes échelles temporelles
- Bass   : Grand hidden state, petite step (slow dynamics)
- Melody : Medium hidden state, medium step
- High   : Petit hidden state, grande step (fast dynamics)
"""

import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import trange

# --------------------------
# Config
# --------------------------
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cpu")

# --------------------------
# Signal de test
# --------------------------
def generate_test_signal(duration=1.0, sr=8000):
    """Signal composite : Bass + Melody + HF"""
    n = int(duration * sr)
    t = np.linspace(0, duration, n, endpoint=False)

    # Bass : sweep 60->100 Hz
    f_bass = np.linspace(60, 100, n)
    phase_bass = 2*np.pi*np.cumsum(f_bass)/sr
    env_bass = 0.6 * (0.5 - 0.5*np.cos(2*np.pi*np.linspace(0, 1, n)))
    bass = env_bass * np.sin(phase_bass)

    # Melody : 350 Hz avec AM 3 Hz
    f_mel = 350.0
    mel = (0.6 + 0.3*np.sin(2*np.pi*3*t)) * np.sin(2*np.pi*f_mel*t)

    # HF : bruit filtré 1-2 kHz
    white = np.random.randn(n)
    hf = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1/sr)
    mask = (freqs >= 1000) & (freqs <= 2000)
    from scipy.signal import get_window
    band = mask.astype(float)
    win = get_window("hann", 33)
    win /= win.sum()
    band = np.convolve(band, win, mode="same")
    hf_filtered = hf * band
    hf_time = np.fft.irfft(hf_filtered, n=n)
    hf_time *= 0.2

    s = bass + mel + hf_time
    return t.astype(np.float32), s.astype(np.float32)


# --------------------------
# RNN Récurrent avec Dynamiques Oscillatoires
# --------------------------
class RecurrentOscillatorRNN(nn.Module):
    """
    RNN récurrent STABILISÉ qui apprend des dynamiques oscillatoires.
    
    Architecture :
        h[t] = (1-α)·h[t-1] + α·tanh(W_hh·h[t-1] + W_xh·x[t] + b_h)
        y[t] = W_hy·h[t] + b_y
    
    Stabilisation via :
    - Leak rate α (< 1) pour empêcher explosion
    - Initialisation orthogonale de W_hh
    - Scaling des poids
    """
    def __init__(self, hidden_size, timescale='medium'):
        super().__init__()
        self.hidden_size = hidden_size
        self.timescale = timescale
        
        # Leak rate selon timescale (crucial pour stabilité !)
        if timescale == 'slow':
            self.alpha = 0.05  # Très lent, peu de mise à jour
        elif timescale == 'fast':
            self.alpha = 0.2   # Rapide
        else:
            self.alpha = 0.1   # Medium
        
        # Matrices récurrentes - INITIALISATION ORTHOGONALE
        self.W_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.W_xh = nn.Parameter(torch.empty(hidden_size, 1))
        self.b_h = nn.Parameter(torch.zeros(hidden_size))
        
        # Projection de sortie - petite initialisation
        self.W_hy = nn.Parameter(torch.empty(1, hidden_size))
        self.b_y = nn.Parameter(torch.zeros(1))
        
        # Initialisation stable
        self._init_stable()
    
    def _init_stable(self):
        """
        Initialisation pour STABILITÉ maximale :
        1. W_hh orthogonale (préserve les normes)
        2. W_xh petite
        3. W_hy petite
        """
        with torch.no_grad():
            # W_hh : Orthogonale avec scaling
            nn.init.orthogonal_(self.W_hh, gain=0.9)
            
            # Ajouter structure oscillatoire sur la diagonale
            for i in range(0, self.hidden_size - 1, 2):
                theta = np.random.uniform(0, 2*np.pi)
                cos_t = np.cos(theta)
                sin_t = np.sin(theta)
                
                # Petit bloc rotatif (rotation faible pour stabilité)
                scale = 0.3  # Très petit pour éviter divergence
                self.W_hh[i:i+2, i:i+2] = torch.tensor([
                    [scale * cos_t, -scale * sin_t],
                    [scale * sin_t,  scale * cos_t]
                ], dtype=torch.float32)
            
            # W_xh : Très petite initialisation
            nn.init.normal_(self.W_xh, mean=0, std=0.01)
            
            # W_hy : Petite initialisation
            nn.init.normal_(self.W_hy, mean=0, std=0.01)
    
    def forward(self, x, h_prev=None):
        """
        Forward pass récurrent STABILISÉ avec leak rate.
        
        Args:
            x : [batch_size, seq_len, 1] ou [seq_len, 1]
            h_prev : État caché précédent [hidden_size] ou None
        
        Returns:
            outputs : [seq_len, 1]
            h_final : [hidden_size]
        """
        # Gérer les dimensions
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [1, seq_len, 1]
        
        batch_size, seq_len, _ = x.shape
        
        # Initialiser l'état caché à ZÉRO (important!)
        if h_prev is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h = h_prev.unsqueeze(0) if h_prev.dim() == 1 else h_prev
        
        outputs = []
        
        # Boucle temporelle avec LEAK RATE
        for t in range(seq_len):
            x_t = x[:, t, :]  # [batch_size, 1]
            
            # Calculer la mise à jour proposée
            h_new = torch.tanh(
                torch.matmul(h, self.W_hh.t()) +
                torch.matmul(x_t, self.W_xh.t()) +
                self.b_h
            )
            
            # LEAK RATE : combinaison de l'ancien et du nouveau
            # h[t] = (1-α)·h[t-1] + α·h_new
            # Ceci empêche l'explosion en gardant une partie de l'ancien état
            h = (1 - self.alpha) * h + self.alpha * h_new
            
            # Sortie
            y_t = torch.matmul(h, self.W_hy.t()) + self.b_y
            outputs.append(y_t)
        
        outputs = torch.cat(outputs, dim=1)  # [batch_size, seq_len, 1]
        return outputs.squeeze(0).squeeze(-1), h.squeeze(0)


# --------------------------
# Modèle Multi-échelle Récurrent
# --------------------------
class MultiScaleRecurrentModel(nn.Module):
    """
    Modèle multi-échelle avec de vrais RNN récurrents.
    
    Chaque bande a son propre RNN avec :
    - Différentes tailles d'état caché
    - Différentes échelles temporelles
    """
    def __init__(self):
        super().__init__()
        
        # Bass : Lent, PETIT état caché pour stabilité
        self.bass_rnn = RecurrentOscillatorRNN(
            hidden_size=24,  # Réduit de 64 à 24
            timescale='slow'
        )
        
        # Melody : Medium, état moyen
        self.melody_rnn = RecurrentOscillatorRNN(
            hidden_size=32,  # Réduit de 80 à 32
            timescale='medium'
        )
        
        # High : Rapide, très petit
        self.high_rnn = RecurrentOscillatorRNN(
            hidden_size=16,  # Réduit de 32 à 16
            timescale='fast'
        )
        
        self.rnns = nn.ModuleList([self.bass_rnn, self.melody_rnn, self.high_rnn])
    
    def forward(self, x):
        """
        x : [seq_len, 1] - Input (peut être zéro pour génération libre)
        
        Returns:
            output : [seq_len] - Signal reconstruit
        """
        outputs = []
        
        for rnn in self.rnns:
            out, _ = rnn(x)
            outputs.append(out)
        
        # Somme des sorties
        return sum(outputs)


# --------------------------
# Baseline (pour comparaison)
# --------------------------
class BaselineOscillators(nn.Module):
    """Oscillateurs Laplace classiques (feed-forward)"""
    def __init__(self, sr=8000, n_units=128):
        super().__init__()
        self.sr = sr
        self.n = n_units
        
        freqs = np.geomspace(20, 3600, num=n_units).astype(np.float32)
        self.register_buffer("freqs", torch.from_numpy(freqs))
        
        self.a = nn.Parameter(torch.randn(self.n) * 0.1)
        self.b = nn.Parameter(torch.randn(self.n) * 0.1)
        
        init_s = np.full((self.n,), 0.25, dtype=np.float32)
        self.log_s = nn.Parameter(torch.from_numpy(np.log(np.expm1(init_s))))
    
    def forward(self, t):
        s = torch.nn.functional.softplus(self.log_s).clamp_max(0.75)
        w = 2*math.pi*self.freqs
        
        t_outer = t[:, None]
        phase = w[None, :] * t_outer
        exp_term = torch.exp(-s[None, :] * t_outer)
        
        y = exp_term * (self.a[None, :]*torch.sin(phase) + self.b[None, :]*torch.cos(phase))
        return y.sum(dim=1)


# --------------------------
# Entraînement
# --------------------------
def train_model(model, s_t, is_recurrent=False, epochs=1000, lr=1e-3, patience=300):
    """Entraîne le modèle avec stabilisation pour RNN"""
    model = model.to(device)
    model.train()
    
    # Learning rate encore plus petit pour RNN
    if is_recurrent:
        lr = 5e-4  # Très prudent
        wd = 1e-4  # Plus de régularisation
    else:
        wd = 1e-5
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()
    
    best_loss = float("inf")
    best_state = None
    no_improve = 0
    
    pbar = trange(epochs, desc="Training", leave=False)
    start = time.time()
    
    # Préparer l'input
    if is_recurrent:
        # Pour RNN : input zéro (génération libre)
        x_input = torch.zeros(len(s_t), 1, device=device)
    else:
        # Pour baseline : temps
        t_t = torch.linspace(0, 1, len(s_t), device=device)
    
    for epoch in pbar:
        optimizer.zero_grad()
        
        if is_recurrent:
            pred = model(x_input)
        else:
            pred = model(t_t)
        
        loss = loss_fn(pred, s_t)
        
        # Vérifier divergence AVANT backward
        if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 10.0:
            print(f"\n   ⚠️ Divergence détectée à epoch {epoch}, loss={loss.item():.6f}")
            break
        
        loss.backward()
        
        # Gradient clipping AGRESSIF pour RNN
        if is_recurrent:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)  # Très strict !
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        cur = loss.item()
        pbar.set_postfix(loss=f"{cur:.6f}")
        
        if cur + 1e-9 < best_loss:
            best_loss = cur
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    
    dur = time.time() - start
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return best_loss, dur


# --------------------------
# Métriques
# --------------------------
def mse(a, b):
    return float(np.mean((a - b) ** 2))

def snr_db(ref, est):
    num = np.sum(ref ** 2) + 1e-12
    den = np.sum((ref - est) ** 2) + 1e-12
    return 10 * np.log10(num / den)


# --------------------------
# Visualisation
# --------------------------
def plot_comparison(t, target, pred_baseline, pred_rnn, sr):
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("RNN Récurrent Multi-échelle vs Baseline Feed-Forward", 
                 fontsize=15, fontweight='bold')
    
    # 1) Waveform zoom
    Tview = int(0.03 * sr)
    axs[0, 0].plot(t[:Tview], target[:Tview], 'k-', label="Target", lw=2, alpha=0.7)
    axs[0, 0].plot(t[:Tview], pred_baseline[:Tview], label="Baseline (FF)", lw=1.5, alpha=0.8)
    axs[0, 0].plot(t[:Tview], pred_rnn[:Tview], label="RNN Récurrent", lw=1.5, alpha=0.8)
    axs[0, 0].set_title("Temporel (0–30 ms)")
    axs[0, 0].set_xlabel("Temps [s]")
    axs[0, 0].set_ylabel("Amplitude")
    axs[0, 0].legend()
    axs[0, 0].grid(alpha=0.3)
    
    # 2) Waveform complet
    axs[0, 1].plot(t, target, 'k-', label="Target", lw=1, alpha=0.5)
    axs[0, 1].plot(t, pred_baseline, label="Baseline (FF)", lw=0.8, alpha=0.8)
    axs[0, 1].plot(t, pred_rnn, label="RNN Récurrent", lw=0.8, alpha=0.8)
    axs[0, 1].set_title("Temporel (complet)")
    axs[0, 1].set_xlabel("Temps [s]")
    axs[0, 1].set_ylabel("Amplitude")
    axs[0, 1].legend()
    axs[0, 1].grid(alpha=0.3)
    
    # 3) Spectre
    def amp_db(x):
        X = np.fft.rfft(x * np.hanning(len(x)))
        A = np.abs(X) + 1e-12
        return 20 * np.log10(A / A.max())
    
    freqs = np.fft.rfftfreq(len(t), 1/sr)
    axs[1, 0].plot(freqs, amp_db(target), 'k-', label="Target", lw=2, alpha=0.7)
    axs[1, 0].plot(freqs, amp_db(pred_baseline), label="Baseline (FF)", lw=1.5, alpha=0.8)
    axs[1, 0].plot(freqs, amp_db(pred_rnn), label="RNN Récurrent", lw=1.5, alpha=0.8)
    axs[1, 0].set_xlim(0, 2000)
    axs[1, 0].set_ylim(-80, 3)
    axs[1, 0].set_title("Spectre (0–2 kHz)")
    axs[1, 0].set_xlabel("Fréquence [Hz]")
    axs[1, 0].set_ylabel("Amplitude [dBFS]")
    axs[1, 0].legend()
    axs[1, 0].grid(alpha=0.3)
    
    # 4) Info
    axs[1, 1].axis('off')
    text = "ARCHITECTURE RNN RÉCURRENT\n"
    text += "=" * 50 + "\n\n"
    text += "Baseline (Feed-Forward) :\n"
    text += "  y(t) = Σ exp(-s·t)·[a·sin(ω·t) + b·cos(ω·t)]\n"
    text += "  → Pas de récurrence\n"
    text += "  → Formule analytique fermée\n\n"
    text += "RNN Récurrent :\n"
    text += "  h[t] = tanh(W_hh·h[t-1] + W_xh·x[t] + b)\n"
    text += "  y[t] = W_hy·h[t] + b\n"
    text += "  → Vraie récurrence temporelle\n"
    text += "  → Réinjection du signal\n"
    text += "  → États cachés persistants\n\n"
    text += "Structure Multi-échelle :\n"
    text += "  • Bass RNN   (64 hidden, slow)\n"
    text += "  • Melody RNN (80 hidden, medium)\n"
    text += "  • High RNN   (32 hidden, fast)\n\n"
    text += "Avantages théoriques :\n"
    text += "  ✓ Dynamiques apprises vs codées\n"
    text += "  ✓ Mémoire temporelle\n"
    text += "  ✓ Corrections itératives possibles\n"
    
    axs[1, 1].text(0.05, 0.98, text, transform=axs[1, 1].transAxes,
                   va='top', ha='left', fontsize=8.5, family='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    return fig


# --------------------------
# Main
# --------------------------
def main():
    sr = 8000
    duration = 1.0
    
    print("="*70)
    print("RNN RÉCURRENT MULTI-ÉCHELLE vs BASELINE FEED-FORWARD")
    print("="*70)
    print(f"\nDevice: {device}\n")
    
    # 1) Signal
    print("1) Génération du signal (1s @ 8kHz)")
    t, s_np = generate_test_signal(duration=duration, sr=sr)
    print(f"   {len(s_np)} échantillons\n")
    
    s_t = torch.from_numpy(s_np).to(device)
    
    # 2) Baseline Feed-Forward
    print("2) Baseline Feed-Forward (128 oscillateurs)")
    model_baseline = BaselineOscillators(sr=sr, n_units=128)
    n_params_b = sum(p.numel() for p in model_baseline.parameters())
    print(f"   Paramètres : {n_params_b}")
    print("   Entraînement...")
    best_b, dur_b = train_model(model_baseline, s_t, is_recurrent=False, 
                                 epochs=1000, lr=5e-3, patience=400)
    print(f"   ✓ Terminé en {dur_b:.1f}s, Loss : {best_b:.6f}\n")
    
    # 3) RNN Récurrent Multi-échelle
    print("3) RNN Récurrent Multi-échelle")
    model_rnn = MultiScaleRecurrentModel()
    n_params_rnn = sum(p.numel() for p in model_rnn.parameters())
    print(f"   Paramètres : {n_params_rnn}")
    print("   Architecture :")
    print("      Bass RNN   : 24 hidden units (slow dynamics)")
    print("      Melody RNN : 32 hidden units (medium dynamics)")
    print("      High RNN   : 16 hidden units (fast dynamics)")
    print("   Entraînement...")
    best_rnn, dur_rnn = train_model(model_rnn, s_t, is_recurrent=True,
                                    epochs=1500, lr=1e-3, patience=500)
    print(f"   ✓ Terminé en {dur_rnn:.1f}s, Loss : {best_rnn:.6f}\n")
    
    # 4) Évaluation
    model_baseline.eval()
    model_rnn.eval()
    
    with torch.no_grad():
        t_t = torch.linspace(0, 1, len(s_t), device=device)
        pred_baseline = model_baseline(t_t).cpu().numpy()
        
        x_input = torch.zeros(len(s_t), 1, device=device)
        pred_rnn = model_rnn(x_input).cpu().numpy()
    
    print("="*70)
    print("RÉSULTATS")
    print("="*70)
    print("\nMétrique             Baseline (FF)   RNN Récurrent   Différence")
    print("-"*70)
    
    mse_b = mse(s_np, pred_baseline)
    mse_rnn = mse(s_np, pred_rnn)
    snr_b = snr_db(s_np, pred_baseline)
    snr_rnn = snr_db(s_np, pred_rnn)
    
    print(f"Paramètres           {n_params_b:<15} {n_params_rnn:<15} {n_params_rnn - n_params_b:+d}")
    print(f"Training Loss        {best_b:<15.6f} {best_rnn:<15.6f} {best_rnn - best_b:+.6f}")
    print(f"MSE                  {mse_b:<15.6f} {mse_rnn:<15.6f} {mse_rnn - mse_b:+.6f}")
    print(f"SNR (dB)             {snr_b:<15.2f} {snr_rnn:<15.2f} {snr_rnn - snr_b:+.2f} dB")
    
    # 5) Visualisation
    print("\n" + "="*70)
    print("VISUALISATION")
    print("="*70)
    fig = plot_comparison(t, s_np, pred_baseline, pred_rnn, sr)
    
    SAVE_DIR = "./outputs"
    os.makedirs(SAVE_DIR, exist_ok=True)
    out_path = os.path.join(SAVE_DIR, "recurrent_vs_feedforward.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Sauvegardé : {out_path}")
    
    # 6) Conclusion
    print("\n" + "="*70)
    print("ANALYSE")
    print("="*70)
    
    if snr_rnn > snr_b:
        print(f"✅ SUCCÈS : Le RNN récurrent surpasse le baseline !")
        print(f"   Amélioration : {snr_rnn - snr_b:.2f} dB")
        print("   → La récurrence aide à capturer les dynamiques")
    elif snr_rnn >= snr_b * 0.95:
        print(f"≈  COMPARABLE : Performance similaire au baseline")
        print(f"   Différence : {snr_rnn - snr_b:.2f} dB")
        print("   → La récurrence n'apporte pas d'avantage significatif")
    else:
        print(f"❌ Le baseline reste meilleur")
        print(f"   Écart : {snr_b - snr_rnn:.2f} dB en faveur du baseline")
        print("   → La récurrence n'aide pas pour ce signal")
        print("   → Les oscillateurs feed-forward sont plus adaptés")

if __name__ == "__main__":
    main()