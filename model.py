#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import torch.nn.functional as F
device = "cuda" if torch.cuda.is_available() else "cpu"

# SMILES Encoder

class SMILESEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, out_dim=2048):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)

        self.cnn1 = nn.Conv1d(embed_dim, embed_dim, 3, padding=1)
        self.cnn2 = nn.Conv1d(embed_dim, embed_dim, 3, padding=1)

        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, out_dim)
        )

    def forward(self, x):
        emb = self.embed(x)
        mask = (x != PAD_IDX).unsqueeze(-1).float()
        emb = emb * mask

        h = emb.transpose(1,2)
        h = F.relu(self.cnn1(h))
        h = F.relu(self.cnn2(h))

        h = (h * mask.transpose(1,2)).sum(2) / mask.sum(1).clamp(min=1)

        z = self.fc(h)
        return F.normalize(z, dim=1)

# Diffusion Scheduler (Stability Optimized)

class DiffusionScheduler:
    def __init__(self, T=200, beta_start=1e-4, beta_end=0.01, device="cpu"):
        """
        Parameter Optimization Notes:
        1. T reduced from 1000 to 200: decreases the number of inference steps, 
        significantly reducing cumulative numerical errors.
        2. beta_end reduced from 0.02 to 0.01: lowers the noise upper bound, 
        preventing excessive disruption of biological features in the late training stage,
        making the noise prediction task more focused and stable.
        """
        self.T = T
        self.device = device

        # Use a linear schedule for parameter update
        self.betas = torch.linspace(
            beta_start, beta_end, T, device=device
        )
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def sample_timesteps(self, n):
        # Randomly sample training steps
        return torch.randint(0, self.T, (n,), device=self.device)

    def add_noise(self, x0, t):
        """
        Forward noising process: q(xt | x0)
        """
        noise = torch.randn_like(x0)
        a_bar = self.alpha_bars[t].unsqueeze(1) # [batch, 1]
        
        # Standard DDPM formula: x_t = sqrt(a_bar) * x0 + sqrt(1 - a_bar) * noise
        xt = torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * noise
        return xt, noise

# Instantiate the scheduler with a more stable 200-step configuration
diffusion = DiffusionScheduler(T=200, beta_start=1e-4, beta_end=0.01, device=device)


# Diffusion Model (Regulatory Operator)

class DiffusionModel(nn.Module):
    def __init__(self, gene_dim, mol_dim=2048, hidden=1024, max_t=200):
        super().__init__()

        # -------- 1. Time Embedding --------
        self.t_embed = nn.Embedding(max_t, hidden)

        # -------- 2. Drug → Gate --------
        # mol_emb (2048) → gene_dim
        self.mol_to_gate = nn.Sequential(
            nn.Linear(mol_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, gene_dim),
            nn.Tanh()
        )

        # -------- 3. Feature encoders --------
        self.fc_x = nn.Linear(gene_dim, hidden)   # noisy Δ
        self.fc_g = nn.Linear(gene_dim, hidden)   # G_mod
        self.fc_t = nn.Linear(hidden, hidden)     # time

        # -------- 4. Denoising Network --------
        self.net = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, gene_dim)
        )

    def forward(self, x_t, G_pre, mol_emb, t):

        # ---- Gate ----
        gate = self.mol_to_gate(mol_emb)

        # ---- G_mod ----
        G_mod = G_pre * gate

        # ---- Encode ----
        hx = self.fc_x(x_t)
        hg = self.fc_g(G_mod)
        ht = self.fc_t(self.t_embed(t))

        h = torch.cat([hx, hg, ht], dim=1)

        return self.net(h)


# ---------- instantiate ----------
model = DiffusionModel(
    gene_dim=len(gene_cols),
    mol_dim=2048,        # Must be consistent with the output of the SMILES encoder
    hidden=1024,
    max_t=diffusion.T
).to(device)

