#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Prediction
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------- 1. File paths --------------------
FINAL_MODEL_PATH = "checkpoints/final_model.pt"
ENCODER_PATH     = "encoder_pretrain_contrastive.pt"
PRED_EXPR_PATH   = "predict_expression.csv"
PRED_MOL_PATH    = "predict_molecules.csv"

# -------------------- 2. Load trained Diffusion Model --------------------
ckpt = torch.load(FINAL_MODEL_PATH, map_location=device)
gene_cols = ckpt["gene_cols"]
Δ_mean = ckpt["Δ_mean"].to(device)
Δ_std  = ckpt["Δ_std"].to(device)
model_cfg = ckpt["model_config"]

class DiffusionModel(nn.Module):
    def __init__(self, gene_dim, mol_dim=2048, hidden=1024, max_t=200):
        super().__init__()
        self.t_embed = nn.Embedding(max_t, hidden)
        self.mol_to_gate = nn.Sequential(
            nn.Linear(mol_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, gene_dim),
            nn.Tanh()
        )
        self.fc_x = nn.Linear(gene_dim, hidden)
        self.fc_g = nn.Linear(gene_dim, hidden)
        self.fc_t = nn.Linear(hidden, hidden)
        self.net = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, gene_dim)
        )

    def forward(self, x_t, G_pre, mol_emb, t):
        gate = self.mol_to_gate(mol_emb)
        G_mod = G_pre * gate
        hx = self.fc_x(x_t)
        hg = self.fc_g(G_mod)
        ht = self.fc_t(self.t_embed(t))
        h = torch.cat([hx, hg, ht], dim=1)
        return self.net(h), gate, G_mod

model = DiffusionModel(
    gene_dim=model_cfg["gene_dim"],
    mol_dim=model_cfg["mol_dim"],
    hidden=model_cfg["hidden"],
    max_t=200
).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

# -------------------- 3. Load trained SMILES Encoder --------------------
encoder_ckpt = torch.load(ENCODER_PATH, map_location=device)
vocab = encoder_ckpt["vocab"]
VOCAB_SIZE = encoder_ckpt["vocab_size"]

PAD_IDX = 0

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
        mask = (x != PAD_IDX).unsqueeze(-1).float()
        emb = self.embed(x) * mask
        h = emb.transpose(1,2)
        h = F.relu(self.cnn1(h))
        h = F.relu(self.cnn2(h))
        h = (h * mask.transpose(1,2)).sum(2) / mask.sum(1).clamp(min=1)
        z = self.fc(h)
        return F.normalize(z, dim=1)

encoder = SMILESEncoder(VOCAB_SIZE, out_dim=model_cfg["mol_dim"]).to(device)
encoder.load_state_dict(encoder_ckpt["encoder_state"])
encoder.eval()

# -------------------- 4. SMILES -> Tensor --------------------
def smiles_to_tensor(smiles, vocab, max_len=128):
    idxs = [vocab.get(c, PAD_IDX) for c in smiles[:max_len]]
    if len(idxs) < max_len:
        idxs += [PAD_IDX]*(max_len-len(idxs))
    return torch.LongTensor(idxs)

# -------------------- 5. Load prediction data --------------------
pred_expr = pd.read_csv(PRED_EXPR_PATH)
pred_mol  = pd.read_csv(PRED_MOL_PATH).set_index("Molecule_id")

# -------------------- 5a. Standardize string type and strip whitespace --------------------
pred_expr["Molecule_id"] = pred_expr["Molecule_id"].astype(str).str.strip()
pred_mol.index = pred_mol.index.astype(str).str.strip()

# -------------------- 5b. Keep only Molecule_id present in pred_mol --------------------
pred_expr = pred_expr[pred_expr["Molecule_id"].isin(pred_mol.index)].reset_index(drop=True)

# -------------------- 6. Simple DDPM reverse sampling prediction --------------------
T = 200 
results = []

print(f"Starting prediction for {len(pred_expr)} samples...")

with torch.no_grad():
    for _, row in tqdm(pred_expr.iterrows(), total=len(pred_expr), desc="Predicting", ncols=100):
        mol_id = row["Molecule_id"]
        if mol_id not in pred_mol.index:
            print(f"Warning: Molecule_id {mol_id} not found in pred_mol, skipping...")
            continue

        G_pre_orig = torch.tensor(row[gene_cols].values.astype("float32")).unsqueeze(0).to(device)
        smiles = pred_mol.loc[mol_id, "Canonical_SMILES"]
        smiles_tensor = smiles_to_tensor(smiles, vocab).unsqueeze(0).to(device)
        mol_emb = encoder(smiles_tensor)

        # Initialize noise
        x = torch.randn_like(G_pre_orig, device=device)

        for t_step in reversed(range(T)):
            t = torch.tensor([t_step], device=device)
            eps_pred, _, _ = model(x, G_pre_orig, mol_emb, t)
            beta_t = 1e-4 + (0.01 - 1e-4) * t_step / (T-1)
            alpha_t = 1 - beta_t
            alpha_bar_t = alpha_t ** (t_step+1)
            coeff = (1 - alpha_t) / np.sqrt(1 - alpha_bar_t)
            mean = (1 / np.sqrt(alpha_t)) * (x - coeff * eps_pred)
            if t_step > 0:
                sigma = np.sqrt(beta_t)
                noise = torch.randn_like(x)
                x = mean + sigma * noise
            else:
                x = mean

        Δ_final = x * Δ_std + Δ_mean
        G_post = torch.clamp(G_pre_orig + Δ_final, min=0.0, max=20.0)

        out = {col: row[col] for col in ["Sample_id","Batch_id","Condition_id","Replicate_id","Molecule_id"]}
        out["Treatment_status"] = "Post"
        G_post_cpu = G_post.squeeze().cpu().numpy()
        for i, g in enumerate(gene_cols):
            out[g] = G_post_cpu[i]

        results.append(out)

# -------------------- 7. save CSV --------------------
pd.DataFrame(results).to_csv("out.csv", index=False)
print("Prediction complete. Saved to out.csv")

