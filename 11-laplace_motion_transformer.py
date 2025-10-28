#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Laplace Motion Transformer - POC
---------------------------------
Combine neurones laplaciens + Transformer pour apprendre des mouvements robotiques.

Les trajectoires sont encodées comme séquences de tokens de mouvement (position, vitesse, accélération),
que le Transformer apprend à prédire de manière autoregressive.

Usage:
    python laplace_motion_transformer.py --trajectory circle --epochs 500
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
import argparse
import math


# ============================================================
# 1. NEURONES LAPLACIENS (encodeur de trajectoire)
# ============================================================

class LaplaceTrajectoryEncoder(nn.Module):
    """
    Encode une trajectoire 3D avec des oscillateurs laplaciens.
    Retourne des features temporelles riches (pos, vel, acc).
    """
    def __init__(self, n_units=64, max_s=0.1, duration=1.0):
        super().__init__()
        self.n_units = n_units
        self.max_s = max_s
        self.duration = duration
        
        # Fréquences harmoniques
        k = torch.arange(1, n_units + 1, dtype=torch.float32)
        omega = 2 * math.pi * k / duration
        self.register_buffer('omega', omega)
        
        # Coefficients pour X, Y, Z
        init_scale = 8.0  # Better initialization for circle with radius 50
        self.a_x = nn.Parameter(torch.randn(n_units) * init_scale)
        self.b_x = nn.Parameter(torch.randn(n_units) * init_scale)
        self.a_y = nn.Parameter(torch.randn(n_units) * init_scale)
        self.b_y = nn.Parameter(torch.randn(n_units) * init_scale)
        self.a_z = nn.Parameter(torch.randn(n_units) * init_scale)
        self.b_z = nn.Parameter(torch.randn(n_units) * init_scale)
        
        # Damping
        self.log_s = nn.Parameter(torch.log(torch.full((n_units,), 0.001)))
        
        # DC offset
        self.dc = nn.Parameter(torch.tensor([0.0, 0.0, 80.0], dtype=torch.float32))
    
    def get_damping(self):
        s = F.softplus(self.log_s)
        return torch.clamp(s, min=1e-9, max=self.max_s)
    
    def forward(self, t):
        """
        t: [B, T] temps
        retourne: [B, T, 3] positions + [B, T, 3] vitesses + [B, T, 3] accélérations
        """
        B, T = t.shape
        
        # Phases
        phase = self.omega.view(1, 1, -1) * t.unsqueeze(2)
        sin_phase = torch.sin(phase)
        cos_phase = torch.cos(phase)
        
        # Damping
        s = self.get_damping()
        decay = torch.exp(-s.view(1, 1, -1) * t.unsqueeze(2))
        
        # Position
        x = (decay * (self.a_x * sin_phase + self.b_x * cos_phase)).sum(dim=2) + self.dc[0]
        y = (decay * (self.a_y * sin_phase + self.b_y * cos_phase)).sum(dim=2) + self.dc[1]
        z = (decay * (self.a_z * sin_phase + self.b_z * cos_phase)).sum(dim=2) + self.dc[2]
        pos = torch.stack([x, y, z], dim=2)  # [B, T, 3]
        
        # Vitesse (dérivée analytique)
        dsin = self.omega.view(1, 1, -1) * cos_phase
        dcos = -self.omega.view(1, 1, -1) * sin_phase
        ddecay = -s.view(1, 1, -1) * decay
        
        vx = ((ddecay * (self.a_x * sin_phase + self.b_x * cos_phase) + 
               decay * (self.a_x * dsin + self.b_x * dcos))).sum(dim=2)
        vy = ((ddecay * (self.a_y * sin_phase + self.b_y * cos_phase) + 
               decay * (self.a_y * dsin + self.b_y * dcos))).sum(dim=2)
        vz = ((ddecay * (self.a_z * sin_phase + self.b_z * cos_phase) + 
               decay * (self.a_z * dsin + self.b_z * dcos))).sum(dim=2)
        vel = torch.stack([vx, vy, vz], dim=2)  # [B, T, 3]
        
        # Accélération (approximation par différences finies)
        dt = 0.01  # timestep approximatif
        acc = torch.zeros_like(vel)
        if T > 2:
            acc[:, 1:-1, :] = (vel[:, 2:, :] - vel[:, :-2, :]) / (2.0 * dt)
            acc[:, 0, :] = (vel[:, 1, :] - vel[:, 0, :]) / dt
            acc[:, -1, :] = (vel[:, -1, :] - vel[:, -2, :]) / dt
        
        return pos, vel, acc


# ============================================================
# 2. MOTION TRANSFORMER (prédit tokens de mouvement)
# ============================================================

class MotionTransformer(nn.Module):
    """
    Transformer autorégressif pour prédire des séquences de mouvements.
    Token = [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, acc_x, acc_y, acc_z]
    """
    def __init__(self, d_model=128, nhead=4, num_layers=3, dim_feedforward=256, 
                 max_seq_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Token = 9 dimensions (3 pos + 3 vel + 3 acc)
        self.token_dim = 9
        
        # Embedding des tokens de mouvement
        self.input_proj = nn.Linear(self.token_dim, d_model)
        
        # Positional encoding
        self.register_buffer('pos_encoding', self._generate_pos_encoding(max_seq_len, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Projection vers tokens de sortie
        self.output_proj = nn.Linear(d_model, self.token_dim)
    
    def _generate_pos_encoding(self, max_len, d_model):
        """Positional encoding sinusoïdal"""
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * 
                            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    def forward(self, motion_tokens, mask=None):
        """
        motion_tokens: [B, T, 9] séquence de tokens de mouvement
        mask: attention mask causal pour autoregression
        retourne: [B, T, 9] prédiction des prochains tokens
        """
        B, T, _ = motion_tokens.shape
        
        # Embedding + positional encoding
        x = self.input_proj(motion_tokens)  # [B, T, d_model]
        x = x + self.pos_encoding[:T, :].unsqueeze(0)
        
        # Causal mask
        if mask is None:
            mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)
        
        # Transformer
        x = self.transformer(x, mask=mask, is_causal=True)
        
        # Projection vers tokens
        out = self.output_proj(x)  # [B, T, 9]
        
        return out


# ============================================================
# 3. GÉNÉRATION DE TRAJECTOIRES SYNTHÉTIQUES
# ============================================================

def generate_trajectory(traj_type, n_points=200, duration=2.0):
    """Génère une trajectoire 3D synthétique"""
    t = np.linspace(0, duration, n_points)

    if traj_type == "circle_flat":
        R = 50.0
        x = R * np.cos(2*np.pi*t/duration)
        y = R * np.sin(2*np.pi*t/duration)
        z = np.full_like(x, 80.0)

    elif traj_type == "circle":  # garde, mais plus soft si tu veux
        x = 50 * np.cos(2 * np.pi * t / duration)
        y = 50 * np.sin(2 * np.pi * t / duration)
        z = 80 + 20 * np.sin(4 * np.pi * t / duration)
    
    elif traj_type == "lemniscate":
        x = 60 * np.cos(2 * np.pi * t / duration) / (1 + np.sin(2 * np.pi * t / duration)**2)
        y = 60 * np.sin(2 * np.pi * t / duration) * np.cos(2 * np.pi * t / duration) / (1 + np.sin(2 * np.pi * t / duration)**2)
        z = 80 + 15 * np.cos(4 * np.pi * t / duration)
    
    elif traj_type == "helix":
        x = 40 * np.cos(4 * np.pi * t / duration)
        y = 40 * np.sin(4 * np.pi * t / duration)
        z = 50 + 60 * (t / duration)
    
    elif traj_type == "star":
        n_branches = 5
        angle = 2 * np.pi * t / duration
        r = 50 * (1 + 0.5 * np.cos(n_branches * angle))
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        z = 80 + 20 * np.sin(3 * angle)
    
    else:
        raise ValueError(f"Unknown trajectory: {traj_type}")
    
    traj = np.stack([x, y, z], axis=1).astype(np.float32)
    return t, traj


# ============================================================
# 4. ENTRAÎNEMENT
# ============================================================

def train_motion_model(trajectories, epochs=500, lr=1e-3, device='cuda'):
    """
    Entraîne seulement l'encodeur Laplacien sur des trajectoires.
    Version simplifiée pour meilleure convergence.
    """
    # Paramètres
    n_traj = len(trajectories)
    duration = 2.0
    n_points = trajectories[0].shape[0]
    
    # Seulement l'encodeur Laplacien
    laplace_encoder = LaplaceTrajectoryEncoder(n_units=512, max_s=0.05, duration=duration).to(device)
    
    # Pas besoin du Transformer pour l'instant
    motion_transformer = None  # Placeholder pour compatibilité
    
    # Optimiseur
    optimizer = torch.optim.AdamW(
        laplace_encoder.parameters(),
        lr=lr, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.001)
    
    # Données
    t = np.linspace(0, duration, n_points, dtype=np.float32)
    t_t = torch.from_numpy(t).unsqueeze(0).to(device).float()
    
    trajectories_t = [torch.from_numpy(traj).unsqueeze(0).to(device).float() for traj in trajectories]
    
    losses = []
    best_loss = float('inf')
    best_state = None
    
    print(f"\n🚀 Entraînement du Laplace Encoder sur {n_traj} trajectoires...")
    print(f"   Laplace Encoder: {sum(p.numel() for p in laplace_encoder.parameters()):,} params\n")
    
    for ep in trange(epochs, desc="Training"):
        total_loss = 0.0
        
        for traj_target in trajectories_t:
            optimizer.zero_grad()
            
            # Encoder génère seulement la position
            pos_pred, vel_pred, acc_pred = laplace_encoder(t_t)
            
            # Loss simple: MSE sur la position uniquement
            loss = F.mse_loss(pos_pred, traj_target)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(laplace_encoder.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / n_traj
        losses.append(avg_loss)
        scheduler.step()
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = laplace_encoder.state_dict()
        
        if (ep + 1) % 100 == 0:
            print(f"\n   Epoch {ep+1}: Loss={avg_loss:.6f}")
    
    print(f"\n✓ Best loss: {best_loss:.6f}")
    
    if best_state is not None:
        laplace_encoder.load_state_dict(best_state)
    
    return laplace_encoder, motion_transformer, losses, t




# ============================================================
# 5. INFÉRENCE AUTOREGRESSIVE
# ============================================================

def generate_motion_sequence(laplace_encoder, motion_transformer, n_steps=200, 
                            temperature=0.5, device='cuda'):
    """
    Génère une séquence de mouvement de manière autoregressive.
    """
    laplace_encoder.eval()
    motion_transformer.eval()
    
    with torch.no_grad():
        # Initialisation : générer premier segment
        t = torch.linspace(0, 0.5, 50, dtype=torch.float32).unsqueeze(0).to(device)
        pos, vel, acc = laplace_encoder(t)
        motion_tokens = torch.cat([pos, vel, acc], dim=2)  # [1, 50, 9]
        
        generated = [motion_tokens]
        
        # Génération autoregressive
        for _ in range(n_steps // 50 - 1):
            # Prédire prochains tokens
            pred = motion_transformer(motion_tokens)
            
            # Ajouter du bruit pour diversité
            pred = pred + torch.randn_like(pred) * temperature
            
            # Prendre les derniers tokens prédits
            next_tokens = pred[:, -10:, :]
            
            # Concaténer
            motion_tokens = torch.cat([motion_tokens, next_tokens], dim=1)
            generated.append(next_tokens)
        
        # Extraire positions uniquement
        all_tokens = torch.cat(generated, dim=1)
        positions = all_tokens[0, :, :3].cpu().numpy()
    
    return positions


# ============================================================
# 6. VISUALISATION 3D + ROBOT ARM
# ============================================================

def ik_6dof_with_z(tx, ty, tz):
    """IK simple pour bras 6-DoF"""
    link_lengths = [40, 35, 30, 25, 20, 15]
    total_height = sum(link_lengths)
    
    cx, cy, cz = 0.0, 0.0, total_height
    positions = [np.array([cx, cy, cz])]
    
    for L in link_lengths:
        dx, dy, dz = tx - cx, ty - cy, tz - cz
        dist = np.linalg.norm([dx, dy, dz])
        if dist < 1e-6:
            dist = 1e-6
        direction = np.array([dx, dy, dz]) / dist
        step = direction * min(L, dist)
        cx += step[0]
        cy += step[1]
        cz += step[2]
        positions.append(np.array([cx, cy, cz]))
    
    return positions


def visualize_motion(trajectory, title="Motion Trajectory"):
    """Visualise une trajectoire avec bras robotique animé"""
    fig = plt.figure(figsize=(16, 7))
    
    # 3D plot avec robot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_xlim(-100, 100)
    ax1.set_ylim(-100, 100)
    ax1.set_zlim(0, 180)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'{title} - Robot Arm', fontsize=14, fontweight='bold')
    ax1.view_init(elev=25, azim=-60)
    
    # Robot arm
    n_joints = 6
    arm_lines = [ax1.plot([], [], [], 'b-', linewidth=6)[0] for _ in range(n_joints)]
    joint_dots = [ax1.plot([], [], [], 'ro', markersize=10)[0] for _ in range(n_joints + 1)]
    
    # Trajectoire cible
    ax1.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
             'gray', alpha=0.3, linewidth=2, label='Target')
    
    # Trajectoire dessinée
    drawing_line, = ax1.plot([], [], [], 'k-', linewidth=3, label='Drawing')
    
    # 2D plots
    ax2 = fig.add_subplot(322)
    ax2.set_title('X-Y Projection', fontsize=12)
    ax2.plot(trajectory[:, 0], trajectory[:, 1], 'gray', alpha=0.5, linewidth=2)
    xy_line, = ax2.plot([], [], 'r-', linewidth=2)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.grid(alpha=0.3)
    ax2.set_aspect('equal')
    
    ax3 = fig.add_subplot(324)
    ax3.set_title('X-Z Projection', fontsize=12)
    ax3.plot(trajectory[:, 0], trajectory[:, 2], 'gray', alpha=0.5, linewidth=2)
    xz_line, = ax3.plot([], [], 'r-', linewidth=2)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.grid(alpha=0.3)
    
    ax4 = fig.add_subplot(326)
    ax4.set_title('Y-Z Projection', fontsize=12)
    ax4.plot(trajectory[:, 1], trajectory[:, 2], 'gray', alpha=0.5, linewidth=2)
    yz_line, = ax4.plot([], [], 'r-', linewidth=2)
    ax4.set_xlabel('Y')
    ax4.set_ylabel('Z')
    ax4.grid(alpha=0.3)
    
    ax1.legend(loc='upper right')
    
    trail = []
    
    def animate(frame):
        idx = frame % len(trajectory)
        tx, ty, tz = trajectory[idx]
        
        # IK
        positions = ik_6dof_with_z(tx, ty, tz)
        pos_array = np.array(positions)
        
        # Update robot arm
        for i, line in enumerate(arm_lines):
            p0, p1 = pos_array[i], pos_array[i + 1]
            line.set_data([p0[0], p1[0]], [p0[1], p1[1]])
            line.set_3d_properties([p0[2], p1[2]])
        
        for i, dot in enumerate(joint_dots):
            x, y, z = pos_array[i]
            dot.set_data([x], [y])
            dot.set_3d_properties([z])
        
        # Update drawing trail
        trail.append([tx, ty, tz])
        if len(trail) > 300:
            trail.pop(0)
        
        if len(trail) > 0:
            tnp = np.array(trail)
            drawing_line.set_data(tnp[:, 0], tnp[:, 1])
            drawing_line.set_3d_properties(tnp[:, 2])
            
            # Update 2D projections
            xy_line.set_data(tnp[:, 0], tnp[:, 1])
            xz_line.set_data(tnp[:, 0], tnp[:, 2])
            yz_line.set_data(tnp[:, 1], tnp[:, 2])
        
        return arm_lines + joint_dots + [drawing_line, xy_line, xz_line, yz_line]
    
    anim = FuncAnimation(fig, animate, frames=len(trajectory) * 3, 
                        interval=20, blit=False, repeat=True)
    plt.tight_layout()
    plt.show()


# ============================================================
# 7. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Laplace Motion Transformer POC")
    parser.add_argument("--trajectory", type=str, default="circle", 
                       choices=["circle", "circle_flat", "lemniscate", "helix", "star"],
                       help="Type de trajectoire d'entraînement")
    parser.add_argument("--n-traj", type=int, default=4,
                       help="Nombre de variations de trajectoires")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--generate", action='store_true',
                       help="Générer une nouvelle séquence après entraînement")
    args = parser.parse_args()
    
    print("="*70)
    print("LAPLACE MOTION TRANSFORMER - POC")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  Device: {device}")
    
    # 1. Générer trajectoires d'entraînement
    print(f"\n📊 Génération de {args.n_traj} trajectoires '{args.trajectory}'...")
    trajectories = []
    for i in range(args.n_traj):
        t, traj = generate_trajectory(args.trajectory, n_points=200, duration=2.0)
        # Ajouter des variations
        noise = np.random.randn(*traj.shape) * 0.5
        traj_var = traj + noise
        trajectories.append(traj_var)
    print(f"   ✓ {len(trajectories)} trajectoires générées")
    
    # 2. Entraînement
    laplace_encoder, motion_transformer, losses, t = train_motion_model(
        trajectories, epochs=args.epochs, lr=args.lr, device=device
    )
    
    # 3. Visualiser la loss
    plt.figure(figsize=(10, 4))
    plt.plot(losses, lw=2)
    plt.yscale('log')
    plt.title('Training Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('motion_transformer_loss.png', dpi=150)
    plt.show()
    print("\n📈 Loss curve saved: motion_transformer_loss.png")
    
    # 4. Test sur trajectoire d'entraînement
    print("\n🎯 Test sur trajectoire d'entraînement...")
    with torch.no_grad():
        t_t = torch.from_numpy(t).unsqueeze(0).to(device).float()
        pos_pred, _, _ = laplace_encoder(t_t)
        traj_pred = pos_pred[0].cpu().numpy()
    
    mse = np.mean((traj_pred - trajectories[0])**2)
    print(f"   MSE: {mse:.6f}")
    
    print("\n🤖 Visualisation avec bras robotique...")
    visualize_motion(traj_pred, title=f"Learned {args.trajectory.capitalize()}")
    
    # 5. Génération autoregressive (optionnel)
    if args.generate:
        print("\n🎲 Génération d'une nouvelle séquence...")
        gen_traj = generate_motion_sequence(
            laplace_encoder, motion_transformer, 
            n_steps=200, temperature=0.3, device=device
        )
        print(f"   ✓ {len(gen_traj)} points générés")
        visualize_motion(gen_traj, title="Generated Motion")
    
    print("\n" + "="*70)
    print("✅ POC terminé !")
    print("="*70)


if __name__ == "__main__":
    main()