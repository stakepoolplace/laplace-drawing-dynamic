#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Transformer Motion Learning - Apprentissage de mouvements robotiques
- Architecture Transformer pour encoder position/vitesse/accélération
- Un token = [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, acc_x, acc_y, acc_z]
- Apprentissage de séquences de mouvements réutilisables
- Capacité à enchainer et interpoler des mouvements appris
- Inférence pour générer de nouvelles variations

Exemples :
    python 13-transformer_motion_learning.py --epochs 2000 --trajectory circle_flat
    python 13-transformer_motion_learning.py --mode inference --load checkpoint.pt
"""

import matplotlib
try:
    matplotlib.use("MacOSX")
except Exception:
    pass

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

import argparse, math, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ANIMS = []

# ----------------------------- Args
p = argparse.ArgumentParser()
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
p.add_argument("--mode", default="train", choices=["train", "inference"])
p.add_argument("--epochs", type=int, default=2000)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--batch_size", type=int, default=32)
p.add_argument("--seq_len", type=int, default=50, help="longueur de séquence pour training")
p.add_argument("--d_model", type=int, default=128, help="dimension du Transformer")
p.add_argument("--nhead", type=int, default=8, help="nombre de têtes d'attention")
p.add_argument("--num_layers", type=int, default=4, help="nombre de couches Transformer")
p.add_argument("--duration", type=float, default=2.0)
p.add_argument("--n_points", type=int, default=400)
p.add_argument("--trajectory", type=str, default="circle_flat",
               choices=["circle_flat","circle_wavy","lemniscate","square","spiral"])
p.add_argument("--seed", type=int, default=123)
p.add_argument("--spin", action="store_true")
p.add_argument("--save", type=str, default="motion_model.pt", help="fichier de sauvegarde")
p.add_argument("--load", type=str, default=None, help="charger un modèle")
args = p.parse_args()

torch.manual_seed(args.seed); np.random.seed(args.seed)
device = torch.device(args.device)
print(f"Device: {device}")

# ----------------------------- Trajectoires cibles
def generate_trajectory(kind="circle_flat", n=400, duration=2.0):
    t = np.linspace(0, duration, n)
    if kind == "circle_flat":
        R = 60.0
        x = R*np.cos(2*np.pi*t/duration)
        y = R*np.sin(2*np.pi*t/duration)
        z = np.full_like(x, 50.0)
    elif kind == "circle_wavy":
        R = 60.0
        x = R*np.cos(2*np.pi*t/duration)
        y = R*np.sin(2*np.pi*t/duration)
        z = 50.0 + 10.0*np.sin(4*np.pi*t/duration)
    elif kind == "lemniscate":
        a = 60.0
        s = np.linspace(-np.pi/2, 3*np.pi/2, n)
        x = a*np.sin(s)/(1+np.cos(s)**2)
        y = a*np.sin(s)*np.cos(s)/(1+np.cos(s)**2)
        z = np.full_like(x, 50.0)
    elif kind == "square":
        # Carré dans le plan XY
        side = 80.0
        points_per_side = n // 4
        x = np.concatenate([
            np.linspace(-side/2, side/2, points_per_side),
            np.full(points_per_side, side/2),
            np.linspace(side/2, -side/2, points_per_side),
            np.full(n - 3*points_per_side, -side/2)
        ])
        y = np.concatenate([
            np.full(points_per_side, -side/2),
            np.linspace(-side/2, side/2, points_per_side),
            np.full(points_per_side, side/2),
            np.linspace(side/2, -side/2, n - 3*points_per_side)
        ])
        z = np.full_like(x, 50.0)
    elif kind == "spiral":
        R_max = 60.0
        R = R_max * t / duration
        x = R * np.cos(4*np.pi*t/duration)
        y = R * np.sin(4*np.pi*t/duration)
        z = 50.0 + 20.0 * t / duration
    else:
        raise ValueError(f"Unknown trajectory: {kind}")
    
    pos = np.stack([x, y, z], axis=-1)
    
    # Calculer vitesse et accélération
    dt = duration / (n - 1)
    vel = np.gradient(pos, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    
    return pos, vel, acc, t

# ----------------------------- Bras 6-DoF (même config)
link_a    = [0.0,  30.0, 25.0,  0.0,  0.0,  0.0]
link_alpha= [np.pi/2, 0.0, 0.0, np.pi/2, -np.pi/2, 0.0]
link_d    = [0.0,   0.0,  0.0, 35.0,    0.0,   15.0]
n_joints = 6

theta_min = torch.tensor([-180,-120,-120,-180,-110,-360], dtype=torch.float32, device=device)*math.pi/180.0
theta_max = torch.tensor([ 180, 120, 120, 180, 110, 360], dtype=torch.float32, device=device)*math.pi/180.0

def dh(a, alpha, d, theta):
    theta = theta.view(-1)
    ct, st = torch.cos(theta), torch.sin(theta)
    a     = torch.as_tensor(a,     device=theta.device, dtype=theta.dtype)
    alpha = torch.as_tensor(alpha, device=theta.device, dtype=theta.dtype)
    d     = torch.as_tensor(d,     device=theta.device, dtype=theta.dtype)
    ca, sa = torch.cos(alpha), torch.sin(alpha)
    a_t  = torch.ones_like(theta) * a
    d_t  = torch.ones_like(theta) * d
    ca_t = torch.ones_like(theta) * ca
    sa_t = torch.ones_like(theta) * sa
    z_t  = torch.zeros_like(theta)
    o_t  = torch.ones_like(theta)
    T = torch.stack([
        torch.stack([ ct,        -st*ca_t,   st*sa_t,  a_t*ct], dim=-1),
        torch.stack([ st,         ct*ca_t,  -ct*sa_t,  a_t*st], dim=-1),
        torch.stack([ z_t,        sa_t,      ca_t,     d_t   ], dim=-1),
        torch.stack([ z_t,        z_t,       z_t,      o_t   ], dim=-1),
    ], dim=-2)
    return T

def fk_thetas(theta_seq):
    """theta_seq: [B,T,6] -> [B,T,3] positions effecteur"""
    if theta_seq.dim() == 2:
        theta_seq = theta_seq.unsqueeze(0)
    B, T, _ = theta_seq.shape
    theta_flat = theta_seq.reshape(-1, 6)  # [B*T, 6]
    
    T_all = []
    for j in range(n_joints):
        Tj = dh(link_a[j], link_alpha[j], link_d[j], theta_flat[:, j])
        T_all.append(Tj)
    Tcum = T_all[0]
    for j in range(1, n_joints):
        Tcum = torch.einsum("bij,bjk->bik", Tcum, T_all[j])
    
    pos = Tcum[:, :3, 3].reshape(B, T, 3)
    return pos.squeeze(0) if B == 1 else pos

# ----------------------------- Architecture Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MotionTransformer(nn.Module):
    """
    Transformer pour encoder et décoder des séquences de mouvements
    Input: [B, T, 9] (pos_xyz + vel_xyz + acc_xyz)
    Output: [B, T, 6] (angles articulaires)
    """
    def __init__(self, d_model=128, nhead=8, num_layers=4, n_joints=6):
        super().__init__()
        self.d_model = d_model
        
        # Encoder les mouvements (9D) vers l'espace latent
        self.motion_embedding = nn.Linear(9, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Decoder vers angles articulaires
        self.joint_decoder = nn.Sequential(
            nn.Linear(d_model, d_model*2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model*2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_joints)
        )
        
    def forward(self, motion_tokens, mask=None):
        """
        motion_tokens: [B, T, 9] - position, vitesse, accélération
        mask: optional attention mask
        Returns: [B, T, 6] - angles articulaires
        """
        # Embedding
        x = self.motion_embedding(motion_tokens)  # [B, T, d_model]
        x = self.pos_encoder(x)
        
        # Transformer
        x = self.transformer(x, mask=mask)  # [B, T, d_model]
        
        # Decode to joint angles
        theta = self.joint_decoder(x)  # [B, T, 6]
        
        return theta
    
    def encode_motion(self, motion_tokens):
        """Encode motion into latent space without decoding"""
        x = self.motion_embedding(motion_tokens)
        x = self.pos_encoder(x)
        return self.transformer(x)

# ----------------------------- Préparation des données
def prepare_motion_tokens(pos, vel, acc):
    """Combiner position, vitesse, accélération en tokens"""
    return np.concatenate([pos, vel, acc], axis=-1)  # [T, 9]

def create_sequences(motion_tokens, seq_len):
    """Créer des séquences glissantes pour l'entraînement"""
    T = motion_tokens.shape[0]
    sequences = []
    for i in range(T - seq_len):
        sequences.append(motion_tokens[i:i+seq_len])
    return np.array(sequences)

# ----------------------------- Training
def train_model(model, motion_data, target_pos, epochs=1000, lr=1e-3, batch_size=32):
    """
    motion_data: [T, 9] - tokens de mouvement
    target_pos: [T, 3] - positions cibles pour FK
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    # Créer des séquences
    sequences = create_sequences(motion_data, args.seq_len)
    target_seqs = create_sequences(target_pos, args.seq_len)
    
    n_seqs = len(sequences)
    print(f"📚 Nombre de séquences d'entraînement: {n_seqs}")
    
    sequences = torch.tensor(sequences, dtype=torch.float32, device=device)
    target_seqs = torch.tensor(target_seqs, dtype=torch.float32, device=device)
    
    print("🚀 Entraînement du Transformer...")
    for epoch in range(1, epochs+1):
        model.train()
        epoch_loss = 0
        n_batches = 0
        
        # Shuffle
        perm = torch.randperm(n_seqs)
        sequences = sequences[perm]
        target_seqs = target_seqs[perm]
        
        for i in range(0, n_seqs, batch_size):
            batch_motion = sequences[i:i+batch_size]  # [B, seq_len, 9]
            batch_target = target_seqs[i:i+batch_size]  # [B, seq_len, 3]
            
            optimizer.zero_grad()
            
            # Forward
            theta_pred = model(batch_motion)  # [B, seq_len, 6]
            
            # FK pour obtenir positions
            pos_pred = fk_thetas(theta_pred)  # [B, seq_len, 3]
            
            # Loss
            pos_loss = F.mse_loss(pos_pred, batch_target)
            
            # Régularisation: limites articulaires
            theta_min_b = theta_min.unsqueeze(0).unsqueeze(0)
            theta_max_b = theta_max.unsqueeze(0).unsqueeze(0)
            over_min = torch.clamp(theta_min_b - theta_pred, min=0.0)
            over_max = torch.clamp(theta_pred - theta_max_b, min=0.0)
            limits_loss = (over_min**2 + over_max**2).mean()
            
            # Smoothness dans le temps
            if theta_pred.size(1) > 1:
                dtheta = theta_pred[:, 1:] - theta_pred[:, :-1]
                smooth_loss = (dtheta**2).mean()
            else:
                smooth_loss = 0.0
            
            loss = pos_loss + 0.001*limits_loss + 0.01*smooth_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        
        if epoch % 100 == 0 or epoch == 1:
            avg_loss = epoch_loss / n_batches
            # Eval
            model.eval()
            with torch.no_grad():
                theta_full = model(sequences[:1])
                pos_full = fk_thetas(theta_full)
                err = torch.sqrt(((pos_full - target_seqs[:1])**2).sum(dim=-1)).mean().item()
            print(f"Epoch {epoch:4d} | Loss: {avg_loss:.5f} | Pos Error: {err:.3f} | LR: {scheduler.get_last_lr()[0]:.6f}")
    
    return model

# ----------------------------- Inference
def inference_motion(model, motion_tokens, seq_len=None):
    """Générer des angles articulaires à partir de tokens de mouvement"""
    model.eval()
    with torch.no_grad():
        if seq_len is None:
            seq_len = motion_tokens.shape[0]
        
        motion_tensor = torch.tensor(motion_tokens, dtype=torch.float32, device=device).unsqueeze(0)
        theta = model(motion_tensor)
        return theta.squeeze(0).cpu().numpy()

# ----------------------------- Main
if __name__ == "__main__":
    # Générer trajectoire
    pos_np, vel_np, acc_np, t_np = generate_trajectory(args.trajectory, args.n_points, args.duration)
    motion_tokens = prepare_motion_tokens(pos_np, vel_np, acc_np)
    
    print(f"📊 Trajectoire '{args.trajectory}':")
    print(f"   Position shape: {pos_np.shape}")
    print(f"   Velocity range: {vel_np.min():.2f} à {vel_np.max():.2f}")
    print(f"   Acceleration range: {acc_np.min():.2f} à {acc_np.max():.2f}")
    print(f"   Motion tokens shape: {motion_tokens.shape}")
    
    # Créer ou charger le modèle
    model = MotionTransformer(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        n_joints=n_joints
    ).to(device)
    
    print(f"🤖 Modèle Transformer: {sum(p.numel() for p in model.parameters())} paramètres")
    
    if args.mode == "train":
        # Training
        model = train_model(model, motion_tokens, pos_np, 
                          epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
        
        # Sauvegarder
        torch.save({
            'model_state': model.state_dict(),
            'args': vars(args),
            'motion_tokens': motion_tokens,
            'pos': pos_np,
        }, args.save)
        print(f"💾 Modèle sauvegardé: {args.save}")
        
        # Inference sur la trajectoire complète
        theta_pred = inference_motion(model, motion_tokens)
        
    elif args.mode == "inference":
        # Charger le modèle
        if args.load is None:
            print("❌ Erreur: spécifiez --load pour le mode inference")
            exit(1)
        
        checkpoint = torch.load(args.load, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        print(f"📂 Modèle chargé: {args.load}")
        
        # Inference
        theta_pred = inference_motion(model, motion_tokens)
    
    # Convertir en torch pour FK
    theta_pred_torch = torch.tensor(theta_pred, dtype=torch.float32, device=device)
    pos_pred = fk_thetas(theta_pred_torch.unsqueeze(0)).squeeze(0).detach().cpu().numpy()
    
    # ----------------------------- Visualisation
    fig = plt.figure(figsize=(14,8))
    ax = fig.add_subplot(121, projection='3d')
    ax.set_title(f"Transformer Motion Learning — {args.trajectory}")
    ax.plot(pos_np[:,0], pos_np[:,1], pos_np[:,2], c='lightgray', lw=2, label="Target")
    ax.plot(pos_pred[:,0], pos_pred[:,1], pos_pred[:,2], c='blue', lw=2, label="Predicted")
    
    # Aspect 1:1:1
    all_pos = np.vstack([pos_np, pos_pred])
    mins = np.min(all_pos, axis=0)
    maxs = np.max(all_pos, axis=0)
    ranges = maxs - mins
    mid = (maxs + mins)/2
    radius = ranges.max()/2 * 1.15
    ax.set_xlim(mid[0]-radius, mid[0]+radius)
    ax.set_ylim(mid[1]-radius, mid[1]+radius)
    ax.set_zlim(mid[2]-radius, mid[2]+radius)
    ax.set_box_aspect((1,1,1))
    ax.legend()
    
    # Projections
    ax_xy = fig.add_subplot(322); ax_xz = fig.add_subplot(324); ax_yz = fig.add_subplot(326)
    ax_xy.plot(pos_np[:,0], pos_np[:,1], c='lightgray'); ax_xy.plot(pos_pred[:,0], pos_pred[:,1], c='blue'); ax_xy.set_title("X-Y")
    ax_xz.plot(pos_np[:,0], pos_np[:,2], c='lightgray'); ax_xz.plot(pos_pred[:,0], pos_pred[:,2], c='blue'); ax_xz.set_title("X-Z")
    ax_yz.plot(pos_np[:,1], pos_np[:,2], c='lightgray'); ax_yz.plot(pos_pred[:,1], pos_pred[:,2], c='blue'); ax_yz.set_title("Y-Z")
    plt.tight_layout()
    
    # ----------------------------- Animation
    def fk_points(theta_frame):
        Tcur = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0)
        pts = [torch.tensor([0,0,0], dtype=torch.float32, device=device)]
        for j in range(n_joints):
            Tj = dh(link_a[j], link_alpha[j], link_d[j], theta_frame[j].unsqueeze(0))
            Tcur = torch.einsum("bij,bjk->bik", Tcur, Tj)
            pts.append(Tcur[0,:3,3])
        return torch.stack(pts, dim=0).detach().cpu().numpy()
    
    fig2 = plt.figure(figsize=(8,8))
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.set_title("Bras — Animation Transformer")
    ax2.set_box_aspect((1,1,1))
    z_min = min(mid[2]-radius, -5)
    ax2.set_xlim(mid[0]-radius, mid[0]+radius)
    ax2.set_ylim(mid[1]-radius, mid[1]+radius)
    ax2.set_zlim(z_min, mid[2]+radius)
    
    arm_lines = [ax2.plot([], [], [], 'b-', lw=5)[0] for _ in range(n_joints)]
    joints = [ax2.plot([], [], [], 'ro', ms=6)[0] for _ in range(n_joints+1)]
    trail, = ax2.plot([], [], [], '-', lw=3, c='blue', alpha=0.85)
    
    xx, yy = np.meshgrid(np.linspace(mid[0]-radius, mid[0]+radius, 10),
                          np.linspace(mid[1]-radius, mid[1]+radius, 10))
    zz = np.zeros_like(xx)
    ax2.plot_surface(xx, yy, zz, alpha=0.1, color='gray')
    
    drawn = []
    T = len(theta_pred)
    
    def init_anim():
        th0 = torch.tensor(theta_pred[0], dtype=torch.float32, device=device)
        pts0 = fk_points(th0)
        for j in range(n_joints):
            p0, p1 = pts0[j], pts0[j+1]
            arm_lines[j].set_data([p0[0], p1[0]], [p0[1], p1[1]])
            arm_lines[j].set_3d_properties([p0[2], p1[2]])
        for j in range(n_joints+1):
            p = pts0[j]
            joints[j].set_data([p[0]], [p[1]])
            joints[j].set_3d_properties([p[2]])
        trail.set_data([pts0[-1][0]], [pts0[-1][1]])
        trail.set_3d_properties([pts0[-1][2]])
        drawn.clear()
        drawn.append(pts0[-1])
        ax2.view_init(elev=25, azim=45)
        return arm_lines + joints + [trail]
    
    def animate(i):
        th = torch.tensor(theta_pred[i % T], dtype=torch.float32, device=device)
        pts = fk_points(th)
        for j in range(n_joints):
            p0, p1 = pts[j], pts[j+1]
            arm_lines[j].set_data([p0[0], p1[0]], [p0[1], p1[1]])
            arm_lines[j].set_3d_properties([p0[2], p1[2]])
        for j in range(n_joints+1):
            p = pts[j]
            joints[j].set_data([p[0]],[p[1]])
            joints[j].set_3d_properties([p[2]])
        ee = pts[-1]
        drawn.append(ee)
        if len(drawn) > 1200: drawn.pop(0)
        arr = np.array(drawn)
        trail.set_data(arr[:,0], arr[:,1]); trail.set_3d_properties(arr[:,2])
        
        if args.spin:
            ax2.view_init(elev=25, azim=(i*0.7)%360)
        
        return arm_lines + joints + [trail]
    
    current_anim = [None]
    
    def start_animation(event):
        try:
            if current_anim[0] is not None:
                current_anim[0].event_source.stop()
        except (AttributeError, TypeError):
            pass
        
        drawn.clear()
        new_anim = FuncAnimation(fig2, animate, frames=T, init_func=init_anim,
                                 interval=20, blit=False, repeat=False)
        current_anim[0] = new_anim
        ANIMS.append(new_anim)
        fig2.canvas.draw_idle()
    
    ax_button = plt.axes([0.81, 0.02, 0.1, 0.05])
    btn_start = Button(ax_button, 'Start')
    btn_start.on_clicked(start_animation)
    
    print("🎬 Animation… Cliquez sur 'Start' pour relancer.")
    start_animation(None)
    plt.show()