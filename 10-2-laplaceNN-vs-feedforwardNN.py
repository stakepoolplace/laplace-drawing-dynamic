#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
==============================================================================
SÉRIES TEMPORELLES · CLASSIFICATION · PROBABILITÉS
Comparaison : Réseau Multi-Bandes Laplaciens vs MLP classique
==============================================================================

NOUVEAU MODÈLE :
  1) CNNFilterBank    : séparation en bandes + débruitage (Conv1d)
  2) LaplacianOscBank : UNE SEULE couche d’oscillateurs, organisée par bandes
                        (amplitude modulée par le contexte passé)
  3) Recomposer       : recomposition (Conv1d 1x1) pour la régression
                        ou tête dense pour la classification / Gauss

Lancement :
  python 11-laplaceNN-vs-feedforwardNN-2.py
"""

from dataclasses import dataclass
from typing import Tuple, Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
from tqdm import trange

# ---------------------------------------------------------------
# Réglages généraux
# ---------------------------------------------------------------
DEVICE = torch.device("cpu")
SEED = 42
rng = np.random.default_rng(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ===============================================================
# MLPs de référence (inchangés)
# ===============================================================

class MLPSeq(nn.Module):  # régression séquentielle
    def __init__(self, ctx_dim: int, hidden=[128,128], out_dim: int = 1):
        super().__init__()
        layers = []
        d = ctx_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.GELU()]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):  # x: [B, L]
        return self.net(x)

class MLPClass(nn.Module):  # classification
    def __init__(self, in_dim: int, hidden=[128,128], n_classes: int = 3):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, n_classes)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# ===============================================================
# NOUVELLE ARCHITECTURE : CNN → Oscillateurs (unique) → Recomposition
# ===============================================================

class CNNFilterBank(nn.Module):
    """
    Sépare le signal passé en bandes de fréquences et réduit le bruit.
    Entrée : [B,1,L]  → Sortie : [B,C,L]
    """
    def __init__(self, in_ch=1, bands=3):
        super().__init__()
        self.bands = bands
        # Trois branches typées (low/mid/high). Pour >3 bandes, dupliquer/adapter.
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
        # Débruitage depthwise + pointwise
        self.denoise = nn.Sequential(
            nn.Conv1d(bands, bands, kernel_size=5, padding=2, groups=bands, bias=False),
            nn.BatchNorm1d(bands), nn.SiLU(),
            nn.Conv1d(bands, bands, kernel_size=1, bias=False)
        )

    def forward(self, x):  # x: [B,1,L]
        low  = self.low(x)
        mid  = self.mid(x)
        high = self.high(x)
        z = torch.cat([low, mid, high], dim=1)  # [B,3,L]
        z = self.denoise(z)                     # [B,3,L]
        return z

class LaplacianOscBank(nn.Module):
    """
    Banque unique d’oscillateurs amortis, organisée par bande.
    y_k(t) = exp(-s_k t) * sin(w_k t + phi_k) * g_k(context)

    - (w_k, s_k, phi_k, gain_k) appris par bande
    - g_k(context) issu d'un encodeur du passé (global avg par bande + Linear)
    """
    def __init__(self, bands=3, K_per_band=32, fs=1000.0,
                 w_min=2*math.pi*0.5, w_max=2*math.pi*120.0,
                 ctx_feat_dim=None):
        super().__init__()
        self.bands = bands
        self.K = K_per_band
        self.fs = fs
        self.w_min = w_min
        self.w_max = w_max

        # Paramètres des oscillateurs
        self.w_raw = nn.Parameter(torch.randn(bands, K_per_band))
        self.s_raw = nn.Parameter(torch.randn(bands, K_per_band) * 0.1)
        self.phase = nn.Parameter(torch.zeros(bands, K_per_band))
        self.gain  = nn.Parameter(torch.ones(bands, K_per_band) * 0.1)

        # Projections contextuelles : pour chaque bande → K gates
        assert ctx_feat_dim is not None, "ctx_feat_dim requis"
        self.ctx_to_gate = nn.ModuleList([
            nn.Sequential(nn.Linear(ctx_feat_dim, K_per_band), nn.Sigmoid())
            for _ in range(bands)
        ])

    def _softplus(self, x):  # s >= 0
        return nn.functional.softplus(x)

    def _w_bounded(self, w_raw):
        return self.w_min + (self.w_max - self.w_min) * torch.sigmoid(w_raw)

    def forward(self, ctx_feat_bandwise, t_fut):
        """
        ctx_feat_bandwise : [B, bands, F]  (features par bande venant du CNN)
        t_fut             : [B, H]         (instants futurs normalisés en secondes)
        Retour            : [B, bands, H]
        """
        B, C, F = ctx_feat_bandwise.shape
        assert C == self.bands
        B2, H = t_fut.shape
        assert B == B2

        # Prépare temps (broadcast)
        t = t_fut.view(B, 1, H)  # [B,1,H]

        outs = []
        for b in range(self.bands):
            # Gates à partir du contexte de la bande b : [B,K]
            g_ctx = self.ctx_to_gate[b](ctx_feat_bandwise[:, b, :])  # [B,K]
            # Paramètres des K oscillateurs de la bande b
            w_b = self._w_bounded(self.w_raw[b]).view(1, self.K, 1)  # [1,K,1]
            s_b = self._softplus(self.s_raw[b]).view(1, self.K, 1)   # [1,K,1]
            phi = self.phase[b].view(1, self.K, 1)                   # [1,K,1]
            g0  = self.gain[b].view(1, self.K, 1)                    # [1,K,1]

            # Base oscillatoire commune pour le batch (broadcast sur B)
            osc = torch.exp(-s_b * t) * torch.sin(w_b * t + phi)     # [B,K,H] via broadcast
            osc = g0 * osc                                           # [B,K,H]

            # Applique gates contextuelles (B,K,1)
            gate = g_ctx.view(B, self.K, 1)
            y_b = (gate * osc).sum(dim=1, keepdim=True)              # [B,1,H]
            outs.append(y_b)

        return torch.cat(outs, dim=1)  # [B,bands,H]

class Recomposer(nn.Module):
    """Mélange 1×1 des bandes pour régression; tête dense pour class/proba."""
    def __init__(self, bands=3, out_mode='regression', out_dim=1):
        super().__init__()
        self.out_mode = out_mode
        if out_mode == 'regression':
            self.mix = nn.Conv1d(bands, 1, kernel_size=1, bias=True)
        elif out_mode == 'classification':
            # On agrège temporellement par bande puis dense → logits
            self.head = nn.Linear(bands, out_dim)
        elif out_mode == 'gauss':
            # Sorties (mu, log_var) sur H pas de temps : on mélange par 1x1 comme en régression,
            # puis une seconde tête 1x1 pour doubler les canaux.
            self.mix_mu    = nn.Conv1d(bands, 1, kernel_size=1, bias=True)
            self.mix_logv  = nn.Conv1d(bands, 1, kernel_size=1, bias=True)
        else:
            raise ValueError("out_mode inconnu")

    def forward(self, y_bands, for_class=False):
        """
        y_bands: [B,bands,H]
        - regression: -> [B,1,H]
        - classification: agrégation temporelle puis dense -> [B,C]
        - gauss: -> (mu, log_var) chacun [B,1,H]
        """
        if self.out_mode == 'regression':
            return self.mix(y_bands)  # [B,1,H]
        elif self.out_mode == 'classification':
            # moyenne temporelle par bande
            feats = y_bands.mean(dim=-1)  # [B,bands]
            return self.head(feats)       # logits [B,C]
        elif self.out_mode == 'gauss':
            mu     = self.mix_mu(y_bands)
            logvar = self.mix_logv(y_bands)
            return mu, logvar

class MultiBandLaplaceSeq(nn.Module):
    """
    Modèle séquentiel (fenêtre passée → sorties futures) avec :
      CNNFilterBank → LaplacianOscBank (unique) → Recomposer
    out_mode ∈ {'regression','classification','gauss'}
    """
    def __init__(self, L: int, H: int,
                 bands=3, K_per_band=32,
                 fs=1000.0,
                 out_mode='regression',
                 out_dim=1):
        super().__init__()
        self.L = L
        self.H = H
        self.bands = bands
        self.frontend = CNNFilterBank(in_ch=1, bands=bands)
        # Contexte: on extrait des features par bande via global avg + stats simples
        self.ctx_pool = nn.AdaptiveAvgPool1d(1)  # par bande → [B,bands,1]
        # dimension de feature par bande (ici 1), on peut ajouter d'autres stats si besoin
        ctx_feat_dim = 1
        self.osc = LaplacianOscBank(bands=bands, K_per_band=K_per_band,
                                    fs=fs, w_min=2*math.pi*0.5, w_max=2*math.pi*120.0,
                                    ctx_feat_dim=ctx_feat_dim)
        self.recomp = Recomposer(bands=bands, out_mode=out_mode, out_dim=out_dim)
        self.out_mode = out_mode

    def forward(self, x_past: torch.Tensor, t_fut: torch.Tensor):
        """
        x_past: [B,L]
        t_fut : [B,H]  (seconds or normalized time steps)
        Retour selon out_mode:
          - 'regression' : [B,H]
          - 'classification' : logits [B,C]
          - 'gauss' : (mu, log_var) chacun [B,H]
        """
        B, L = x_past.shape
        assert L == self.L
        # Frontend CNN sur la fenêtre passée
        x = x_past.view(B,1,L)                 # [B,1,L]
        bands_sig = self.frontend(x)           # [B,bands,L]
        # Features de contexte par bande (global avg)
        ctx = self.ctx_pool(bands_sig).squeeze(-1)  # [B,bands]

        # Prépare features au format [B,bands,F]
        ctx_feat = ctx.unsqueeze(-1)           # F=1 → [B,bands,1]

        # Oscillateurs → sorties par bande sur l’horizon futur
        y_bands = self.osc(ctx_feat, t_fut)    # [B,bands,H]

        if self.out_mode == 'regression':
            y = self.recomp(y_bands).squeeze(1)     # [B,H]
            return y
        elif self.out_mode == 'classification':
            logits = self.recomp(y_bands)           # [B,C]
            return logits
        elif self.out_mode == 'gauss':
            mu, log_var = self.recomp(y_bands)      # [B,1,H] chacun
            return mu.squeeze(1), log_var.squeeze(1)
        else:
            raise ValueError("out_mode inconnu")

# ===============================================================
# Générateurs de datasets (inchangés)
# ===============================================================

def make_windows(x: np.ndarray, L: int, H: int) -> Tuple[np.ndarray, np.ndarray]:
    """Transforme une série x en (X, Y) avec fenêtre L et horizon H (multi-step)."""
    N = len(x) - L - H + 1
    X = np.stack([x[i:i+L] for i in range(N)], axis=0)
    Y = np.stack([x[i+L:i+L+H] for i in range(N)], axis=0)
    return X.astype(np.float32), Y.astype(np.float32)

# A1) AR(1)
@dataclass
class AR1Params:
    phi: float = 0.8
    sigma: float = 0.2

def gen_ar1(T=2000, phi=0.8, sigma=0.2) -> np.ndarray:
    y = np.zeros(T, dtype=np.float32)
    for t in range(1, T):
        y[t] = phi*y[t-1] + rng.normal(0, sigma)
    return y

# A2) Saisonnier + tendance
def gen_seasonal(T=2000, f=1/50, trend=0.001, sigma=0.2) -> np.ndarray:
    t = np.arange(T)
    y = 0.8*np.sin(2*np.pi*f*t) + trend*t + rng.normal(0, sigma, size=T)
    return y.astype(np.float32)

# A3) Changement de régime (switch AR)
def gen_switch_ar(T=2000, phi1=0.2, phi2=0.95, p_switch=0.01, sigma=0.15) -> np.ndarray:
    y = np.zeros(T, dtype=np.float32)
    phi = phi1
    for t in range(1, T):
        if rng.random() < p_switch:
            phi = phi2 if abs(phi - phi1) < 1e-6 else phi1
        y[t] = phi*y[t-1] + rng.normal(0, sigma)
    return y

# B1) Classification : 3 familles de signaux
def gen_family_signal(T=256, family=0) -> np.ndarray:
    t = np.linspace(0, 1, T, dtype=np.float32)
    if family == 0:  # sin
        f = rng.uniform(3, 6)
        y = np.sin(2*np.pi*f*t)
    elif family == 1:  # chirp
        f0, f1 = rng.uniform(2,4), rng.uniform(6,10)
        k = (f1-f0)
        phase = 2*np.pi*(f0*t + 0.5*k*t*t)
        y = np.sin(phase)
    else:  # square-like (harmoniques impairs)
        y = np.zeros_like(t)
        base = rng.uniform(2, 5)
        for k in range(1, 10, 2):
            y += (1.0/k)*np.sin(2*np.pi*k*base*t)
        y *= (4/np.pi)
    y += rng.normal(0, 0.05, size=T)
    return y.astype(np.float32)

def make_class_dataset(N=3000, T=256, L=128) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for _ in range(N):
        c = rng.integers(0, 3)
        sig = gen_family_signal(T=T, family=int(c))
        X.append(sig[:L])
        y.append(c)
    return np.stack(X).astype(np.float32), np.array(y, dtype=np.int64)

# B2) Détection d'événement rare (spike)
def gen_spike_series(T=256, p_spike=0.2) -> Tuple[np.ndarray, int]:
    y = rng.normal(0, 0.2, size=T).astype(np.float32)
    label = 0
    if rng.random() < p_spike:
        pos = rng.integers(T//4, 3*T//4)
        y[pos:pos+3] += rng.normal(3.0, 0.2)
        label = 1
    return y, label

def make_spike_dataset(N=3000, L=128) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for _ in range(N):
        sig, lab = gen_spike_series()
        X.append(sig[:L])
        y.append(lab)
    return np.stack(X).astype(np.float32), np.array(y, dtype=np.int64)

# C1) Probabiliste : bruit hétéroscédastique (variance dépend du temps)
def gen_hetero(T=2000) -> np.ndarray:
    t = np.linspace(0, 1, T)
    mean = np.sin(2*np.pi*3*t)
    std = 0.05 + 0.35*(t**2)  # bruit augmente dans le temps
    y = mean + rng.normal(0, std)
    return y.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)

# ===============================================================
# Entraînement générique (adapté au nouveau modèle)
# ===============================================================

def train_regressor_seq(model, Xtr, Ytr, epochs=50, lr=1e-3, batch=64):
    model.to(DEVICE).train()
    opt = optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    N = len(Xtr)
    for _ in trange(epochs, leave=False):
        idx = torch.randperm(N)
        for i in range(0, N, batch):
            b = idx[i:i+batch]
            xb = torch.from_numpy(Xtr[b]).to(DEVICE)  # [B, L]
            yb = torch.from_numpy(Ytr[b]).to(DEVICE)  # [B, H]
            opt.zero_grad()
            B, H = yb.shape
            # Temps futurs uniformes normalisés dans [0,1]
            t = torch.linspace(0, 1, H, dtype=torch.float32, device=DEVICE).repeat(B,1)
            if isinstance(model, MultiBandLaplaceSeq):
                pred = model(xb, t)                 # [B,H]
            else:  # MLPSeq prédit H valeurs d'un coup à partir du contexte
                pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

def eval_regressor_seq(model, Xte, Yte) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(Xte).to(DEVICE)
        yb = torch.from_numpy(Yte).to(DEVICE)
        B, H = yb.shape
        t = torch.linspace(0, 1, H, dtype=torch.float32, device=DEVICE).repeat(B,1)
        if isinstance(model, MultiBandLaplaceSeq):
            pred = model(xb, t)
        else:
            pred = model(xb)
        mse = nn.MSELoss()(pred, yb).item()
        mae = nn.L1Loss()(pred, yb).item()
        rmse = float(np.sqrt(mse))
    return {"MSE": mse, "MAE": mae, "RMSE": rmse}

def train_classifier(model, Xtr, ytr, epochs=30, lr=1e-3, batch=128):
    model.to(DEVICE).train()
    opt = optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    N = len(Xtr)
    for _ in trange(epochs, leave=False):
        idx = torch.randperm(N)
        for i in range(0, N, batch):
            b = idx[i:i+batch]
            xb = torch.from_numpy(Xtr[b]).to(DEVICE)
            yb = torch.from_numpy(ytr[b]).to(DEVICE)
            opt.zero_grad()
            if isinstance(model, MultiBandLaplaceSeq):
                # on fournit un H=1 arbitraire pour générer un état temporel synthétique
                t = torch.ones((xb.shape[0], 1), device=DEVICE)
                logits = model(xb, t)
            else:
                logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

def eval_classifier(model, Xte, yte) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(Xte).to(DEVICE)
        yb = torch.from_numpy(yte).to(DEVICE)
        if isinstance(model, MultiBandLaplaceSeq):
            t = torch.ones((xb.shape[0], 1), device=DEVICE)
            logits = model(xb, t)
        else:
            logits = model(xb)
        probs = nn.functional.softmax(logits, dim=-1)
        pred = probs.argmax(dim=-1)
        acc = (pred == yb).float().mean().item()
        # Brier score (multi-classe)
        y_onehot = nn.functional.one_hot(yb, num_classes=probs.shape[-1]).float()
        brier = torch.mean((probs - y_onehot)**2).item()
    return {"Acc": acc, "Brier": brier}

# Probabiliste (régression Gaussienne)
def nll_gaussian(mu, log_var, y):
    return 0.5*(log_var + (y - mu)**2 / torch.exp(log_var))

def train_gauss(model, Xtr, Ytr, epochs=50, lr=1e-3, batch=64):
    model.to(DEVICE).train()
    opt = optim.AdamW(model.parameters(), lr=lr)
    N = len(Xtr)
    for _ in trange(epochs, leave=False):
        idx = torch.randperm(N)
        for i in range(0, N, batch):
            b = idx[i:i+batch]
            xb = torch.from_numpy(Xtr[b]).to(DEVICE)
            yb = torch.from_numpy(Ytr[b]).to(DEVICE)
            B, H = yb.shape
            t = torch.linspace(0, 1, H, dtype=torch.float32, device=DEVICE).repeat(B,1)
            opt.zero_grad()
            if isinstance(model, MultiBandLaplaceSeq):
                mu, log_var = model(xb, t)
            else:
                mu, log_var = model(xb)
            log_var = torch.clamp(log_var, min=-6.0, max=6.0)
            loss = nll_gaussian(mu, log_var, yb).mean()
            loss.backward()
            opt.step()

def eval_gauss(model, Xte, Yte) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(Xte).to(DEVICE)
        yb = torch.from_numpy(Yte).to(DEVICE)
        B, H = yb.shape
        t = torch.linspace(0, 1, H, dtype=torch.float32, device=DEVICE).repeat(B,1)
        if isinstance(model, MultiBandLaplaceSeq):
            mu, log_var = model(xb, t)
        else:
            mu, log_var = model(xb)
        log_var = torch.clamp(log_var, min=-6.0, max=6.0)
        nll = nll_gaussian(mu, log_var, yb).mean().item()
        sigma = torch.exp(0.5*log_var)
        lo, hi = mu - 1.64*sigma, mu + 1.64*sigma
        picp = ((yb >= lo) & (yb <= hi)).float().mean().item()
    return {"NLL": nll, "PICP90": picp}

# ===============================================================
# Batteries de tests (identiques, modèles remplacés)
# ===============================================================

def test_A_timeseries():
    print("\n=== A) PRÉVISION SÉRIE TEMPORELLE ===")
    L, H = 64, 16
    # AR(1)
    y = gen_ar1(T=4000, phi=0.85, sigma=0.25)
    X, Y = make_windows(y, L=L, H=H)
    n = int(0.8*len(X))
    Xtr, Ytr, Xte, Yte = X[:n], Y[:n], X[n:], Y[n:]
    # Modèles
    osc = MultiBandLaplaceSeq(L=L, H=H, bands=3, K_per_band=32, fs=1000.0,
                              out_mode='regression', out_dim=H)
    mlp = MLPSeq(ctx_dim=L, hidden=[128,128], out_dim=H)
    # Train + Eval
    train_regressor_seq(osc, Xtr, Ytr, epochs=60, lr=3e-3)
    train_regressor_seq(mlp, Xtr, Ytr, epochs=60, lr=1e-3)
    e_osc = eval_regressor_seq(osc, Xte, Yte)
    e_mlp = eval_regressor_seq(mlp, Xte, Yte)
    print("AR(1) → MB-OSC:", e_osc, " | MLP:", e_mlp)

    # Saisonnier + tendance
    y = gen_seasonal(T=4000, f=1/50, trend=0.0008, sigma=0.25)
    X, Y = make_windows(y, L=L, H=H)
    n = int(0.8*len(X))
    Xtr, Ytr, Xte, Yte = X[:n], Y[:n], X[n:], Y[n:]
    osc2 = MultiBandLaplaceSeq(L=L, H=H, bands=3, K_per_band=32, fs=1000.0,
                               out_mode='regression', out_dim=H)
    mlp2 = MLPSeq(ctx_dim=L, hidden=[128,128], out_dim=H)
    train_regressor_seq(osc2, Xtr, Ytr, epochs=60, lr=3e-3)
    train_regressor_seq(mlp2, Xtr, Ytr, epochs=60, lr=1e-3)
    e_osc2 = eval_regressor_seq(osc2, Xte, Yte)
    e_mlp2 = eval_regressor_seq(mlp2, Xte, Yte)
    print("Seasonal+Trend → MB-OSC:", e_osc2, " | MLP:", e_mlp2)

    # Switch AR
    y = gen_switch_ar(T=4000, phi1=0.2, phi2=0.95, p_switch=0.02, sigma=0.18)
    X, Y = make_windows(y, L=L, H=H)
    n = int(0.8*len(X))
    Xtr, Ytr, Xte, Yte = X[:n], Y[:n], X[n:], Y[n:]
    osc3 = MultiBandLaplaceSeq(L=L, H=H, bands=3, K_per_band=32, fs=1000.0,
                               out_mode='regression', out_dim=H)
    mlp3 = MLPSeq(ctx_dim=L, hidden=[128,128], out_dim=H)
    train_regressor_seq(osc3, Xtr, Ytr, epochs=60, lr=3e-3)
    train_regressor_seq(mlp3, Xtr, Ytr, epochs=60, lr=1e-3)
    e_osc3 = eval_regressor_seq(osc3, Xte, Yte)
    e_mlp3 = eval_regressor_seq(mlp3, Xte, Yte)
    print("Switch-AR → MB-OSC:", e_osc3, " | MLP:", e_mlp3)

def test_B_classification():
    print("\n=== B) CLASSIFICATION DE SÉQUENCES ===")
    # Multi-classe (3 familles)
    X, y = make_class_dataset(N=4000, T=256, L=128)
    n = int(0.8*len(X))
    Xtr, ytr, Xte, yte = X[:n], y[:n], X[n:], y[n:]
    # Modèles
    osc = MultiBandLaplaceSeq(L=128, H=1, bands=3, K_per_band=32, fs=1000.0,
                              out_mode='classification', out_dim=3)
    mlp = MLPClass(in_dim=128, hidden=[128,128], n_classes=3)

    # Entraînement (H=1 arbitraire pour le temps futur)
    def train_class_with_mb(model, Xtr, ytr, epochs=35, lr=2e-3, batch=128):
        model.to(DEVICE).train()
        opt = optim.AdamW(model.parameters(), lr=lr)
        N = len(Xtr)
        for _ in trange(epochs, leave=False):
            idx = torch.randperm(N)
            for i in range(0, N, batch):
                b = idx[i:i+batch]
                xb = torch.from_numpy(Xtr[b]).to(DEVICE)
                yb = torch.from_numpy(ytr[b]).to(DEVICE)
                t = torch.ones((xb.shape[0], 1), device=DEVICE)  # H=1
                opt.zero_grad()
                logits = model(xb, t)
                loss = nn.CrossEntropyLoss()(logits, yb)
                loss.backward()
                opt.step()

    train_class_with_mb(osc, Xtr, ytr)
    train_classifier(mlp, Xtr, ytr)

    # Évaluation
    e_osc = eval_classifier(osc, Xte, yte)
    e_mlp = eval_classifier(mlp, Xte, yte)
    print("Familles → MB-OSC:", e_osc, " | MLP:", e_mlp)

    # Binaire (spike)
    X, y = make_spike_dataset(N=4000, L=128)
    n = int(0.8*len(X))
    Xtr, ytr, Xte, yte = X[:n], y[:n], X[n:], y[n:]
    osc2 = MultiBandLaplaceSeq(L=128, H=1, bands=3, K_per_band=32, fs=1000.0,
                               out_mode='classification', out_dim=2)
    mlp2 = MLPClass(in_dim=128, hidden=[128,128], n_classes=2)

    train_class_with_mb(osc2, Xtr, ytr, epochs=30)
    train_classifier(mlp2, Xtr, ytr, epochs=30)

    # Eval (ECE simple)
    def ece_score(probs, labels, n_bins=10):
        conf, pred = probs.max(dim=-1)
        bins = torch.linspace(0,1,n_bins+1, device=probs.device)
        ece = torch.tensor(0., device=probs.device)
        for i in range(n_bins):
            m = (conf>bins[i]) & (conf<=bins[i+1])
            if m.any():
                acc = (pred[m]==labels[m]).float().mean()
                ece += (m.float().mean()) * torch.abs(acc - conf[m].mean())
        return ece.item()

    def eval_bin(model, X, y):
        model.eval()
        with torch.no_grad():
            xb = torch.from_numpy(X).to(DEVICE)
            yb = torch.from_numpy(y).to(DEVICE)
            t = torch.ones((xb.shape[0], 1), device=DEVICE)
            logits = model(xb, t) if isinstance(model, MultiBandLaplaceSeq) else model(xb)
            probs = nn.functional.softmax(logits, dim=-1)
            pred = probs.argmax(dim=-1)
            acc = (pred==yb).float().mean().item()
            ece = ece_score(probs, yb)
            brier = torch.mean((probs[:,1]-yb.float())**2).item()
        return {"Acc": acc, "ECE": ece, "Brier": brier}

    e_osc2 = eval_bin(osc2, Xte, yte)
    e_mlp2 = eval_bin(mlp2, Xte, yte)
    print("Spike-det → MB-OSC:", e_osc2, " | MLP:", e_mlp2)

def test_C_probabilistic():
    print("\n=== C) SORTIES PROBABILISTES (Gauss) ===")
    L, H = 64, 8
    y, mean, std = gen_hetero(T=4000)
    X, Y = make_windows(y, L=L, H=H)
    n = int(0.8*len(X))
    Xtr, Ytr, Xte, Yte = X[:n], Y[:n], X[n:], Y[n:]
    osc = MultiBandLaplaceSeq(L=L, H=H, bands=3, K_per_band=32, fs=1000.0,
                              out_mode='gauss', out_dim=H)
    train_gauss(osc, Xtr, Ytr, epochs=70, lr=2e-3)
    e = eval_gauss(osc, Xte, Yte)
    print("Hétéroscédastique → MB-OSC:", e)

    # Baseline MLP Gauss
    class MLPGauss(MLPSeq):
        def __init__(self, ctx_dim, hidden, out_dim):
            super().__init__(ctx_dim, hidden, out_dim*2)
        def forward(self, x):
            out = super().forward(x)
            mu, log_var = torch.chunk(out, 2, dim=-1)
            return mu, log_var
    mlp = MLPGauss(ctx_dim=L, hidden=[128,128], out_dim=H)
    train_gauss(mlp, Xtr, Ytr, epochs=70, lr=1e-3)
    # Eval
    mlp.eval()
    with torch.no_grad():
        xb = torch.from_numpy(Xte).to(DEVICE)
        yb = torch.from_numpy(Yte).to(DEVICE)
        mu, log_var = mlp(xb)
        log_var = torch.clamp(log_var, min=-6.0, max=6.0)
        nll = nll_gaussian(mu, log_var, yb).mean().item()
        sigma = torch.exp(0.5*log_var)
        lo, hi = mu - 1.64*sigma, mu + 1.64*sigma
        picp = ((yb >= lo) & (yb <= hi)).float().mean().item()
    print("Hétéroscédastique → MLP:", {"NLL": nll, "PICP90": picp})

# ===============================================================
# Main
# ===============================================================

def main():
    test_A_timeseries()
    test_B_classification()
    test_C_probabilistic()
    print("\nTerminé.")

if __name__ == "__main__":
    main()
