#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
device = "cuda" if torch.cuda.is_available() else "cpu"

# Data Loading
expr_df = pd.read_csv("expression.csv")
mol_df  = pd.read_csv("molecules.csv")

meta_cols = [
    "Sample_id", "Batch_id", "Condition_id",
    "Replicate_id", "Molecule_id", "Treatment_status"
]

gene_cols = [c for c in expr_df.columns if c not in meta_cols]

expr_df[gene_cols] = expr_df[gene_cols].apply(
    pd.to_numeric, errors="coerce"
).fillna(0.0)

mol2smiles = dict(
    zip(mol_df["Molecule_id"], mol_df["Canonical_SMILES"])
)

# S1 + S2 for encoder
all_smiles = mol_df["Canonical_SMILES"].tolist()

# S2 ids
s2_ids = mol_df[mol_df["Source"] == "S2"]["Molecule_id"].unique()

expr_mol_ids = set(expr_df["Molecule_id"].unique())

s2_expr_ids = sorted(
    set(mol_df[mol_df["Source"] == "S2"]["Molecule_id"])
    & expr_mol_ids
)

# 5-Fold CV (expression-aware)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
s2_expr_ids = np.array(s2_expr_ids)

fold_models = []

for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(s2_expr_ids)):
    print(f"\n=== Fold {fold_idx+1} / 5 ===")

    train_s2 = set(s2_expr_ids[train_idx])
    val_s2   = set(s2_expr_ids[val_idx])

    print(
        f"Train mols: {len(train_s2)} | Val mols: {len(val_s2)}"
    )

# Build Δ using Pre/Post matching (Batch/Replicate DMSO)
import os
pairs_file = f"pairs_fold{fold_idx}.pkl"
delta_norm_file = f"delta_norm_fold{fold_idx}.pt"

# If a saved model exists, load it directly
if os.path.exists(pairs_file) and os.path.exists(delta_norm_file):
    print("[Stage6] Found cached pairs and delta_norm. Loading...")
    import pickle
    with open(pairs_file, "rb") as f:
        pairs = pickle.load(f)
    delta_norm = torch.load(delta_norm_file)
else:
    print("[Stage6] Building Δ pairs from Pre/Post matching...")
    pairs = []

    for mid in tqdm(train_s2, desc="[Stage6] Molecules"):
        # Find post-treatment samples for the current molecule
        post_df = expr_df[
            (expr_df["Molecule_id"] == mid) & 
            (expr_df["Treatment_status"] == "Post")
        ]
        if len(post_df) == 0:
            continue  # Skip if no post-treatment samples are available

        for _, post_row in post_df.iterrows():
            batch_id = post_row["Batch_id"]
            rep_id   = post_row["Replicate_id"]

            # Find the corresponding batch/replicate DMSO pre-treatment samples
            pre_df = expr_df[
                (expr_df["Molecule_id"] == "DMSO") &
                (expr_df["Treatment_status"] == "Pre") &
                (expr_df["Batch_id"] == batch_id) &
                (expr_df["Replicate_id"] == rep_id)
            ]

            if len(pre_df) == 0:
                # If no DMSO pre-treatment samples exist for this plate, use the batch-level DMSO pre-treatment mean
                pre_df = expr_df[
                    (expr_df["Molecule_id"] == "DMSO") &
                    (expr_df["Treatment_status"] == "Pre") &
                    (expr_df["Batch_id"] == batch_id)
                ]

            if len(pre_df) == 0:
                # If still unavailable, skip this post-treatment sample
                print(f"Skipping Post sample {post_row['Sample_id']}: no DMSO Pre in batch {batch_id}")
                continue

            # Compute the mean of G_pre
            G_pre_mean = pre_df[gene_cols].values.astype("float32").mean(axis=0)

            # Expression values of the current post-treatment sample
            G_post_vals = post_row[gene_cols].values.astype("float32")

            # Δ = Post - Pre_mean
            Δ = G_post_vals - G_pre_mean

            # Add to pairs
            pairs.append({
                "G_pre": G_pre_mean,
                "Δ": Δ,
                "Molecule_id": mid,
                "Condition_id": post_row["Condition_id"]
            })

    # Check Results
    if len(pairs) == 0:
        raise ValueError("No Δ pairs found! Please check your Pre/Post data and batch matching.")
    else:
        print(f"Built {len(pairs)} Δ pairs using Batch/Replicate DMSO Pre matching.")

    # Compute delta_norm
    Δ_all = torch.tensor(
        np.stack([p["Δ"] for p in pairs]),
        dtype=torch.float32
    )

    Δ_mean = Δ_all.mean(dim=0)
    Δ_std  = Δ_all.std(dim=0)
    Δ_std  = torch.clamp(Δ_std, min=0.1)

    delta_norm = {"Δ_mean": Δ_mean, "Δ_std": Δ_std}

    # Save pairs and delta_norm
    import pickle
    with open(pairs_file, "wb") as f:
        pickle.dump(pairs, f)
    torch.save(delta_norm, delta_norm_file)
    print(f"[Stage6] Saved pairs to {pairs_file} and delta_norm to {delta_norm_file}")

# Print Δ_std statistics
print(
    f"Δ_std stats: max={delta_norm['Δ_std'].max().item():.4f}, "
    f"min={delta_norm['Δ_std'].min().item():.4f}, "
    f"mean={delta_norm['Δ_std'].mean().item():.4f}"
)

# Training (Stable Noise Prediction)
print("Stage 9: Training Diffusion Model")

# ---------------- Load pretrained encoder ----------------
ckpt = torch.load("encoder_pretrain_contrastive.pt", map_location=device)

encoder = SMILESEncoder(VOCAB_SIZE).to(device)
encoder.load_state_dict(ckpt["encoder_state"])
encoder.eval()

for p in encoder.parameters():
    p.requires_grad = False

# ---------------- Pearson ----------------
def pearson_corr(x, y, eps=1e-8):
    x_mean = x.mean(dim=1, keepdim=True)
    y_mean = y.mean(dim=1, keepdim=True)
    xm = x - x_mean
    ym = y - y_mean
    r_num = (xm * ym).sum(dim=1)
    r_den = torch.sqrt((xm**2).sum(dim=1) * (ym**2).sum(dim=1) + eps)
    return (r_num / r_den).mean()

# ---------------- Dataset ----------------
class DeltaDataset(Dataset):
    def __init__(self, pairs, mol2smiles, vocab, delta_norm):
        self.pairs = pairs
        self.mol2smiles = mol2smiles
        self.vocab = vocab
        self.Δ_mean = delta_norm["Δ_mean"]
        self.Δ_std  = delta_norm["Δ_std"]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        G_pre = torch.tensor(p["G_pre"], dtype=torch.float32)
        Δ     = torch.tensor(p["Δ"], dtype=torch.float32)
        Δ_norm = (Δ - self.Δ_mean) / self.Δ_std
        smiles = self.mol2smiles[p["Molecule_id"]]
        smiles_tensor = smiles_to_tensor(smiles, self.vocab)
        return {"G_pre": G_pre, "Δ": Δ_norm, "smiles": smiles_tensor}

dataset = DeltaDataset(pairs, mol2smiles, vocab, delta_norm)
loader = DataLoader(dataset, batch_size=32, shuffle=True, pin_memory=(device == "cuda"))

# ---------------- Model ----------------
model = DiffusionModel(
    gene_dim=len(gene_cols),
    mol_dim=2048,
    hidden=1024,
    max_t=diffusion.T
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ---------------- Training ----------------
λ_noise = 1.0
λ_corr  = 0.5
λ_scale = 0.1

loss_total_history = []
loss_noise_history = []
loss_scale_history = []
loss_corr_history  = []

num_epochs = 300

Δ_mean_device = delta_norm["Δ_mean"].to(device)
Δ_std_device  = delta_norm["Δ_std"].to(device)

for epoch in range(num_epochs):

    model.train()

    total_loss_epoch = 0
    noise_epoch = 0
    scale_epoch = 0
    corr_epoch  = 0

    for batch in loader:

        G_pre  = batch["G_pre"].to(device)
        Δ_true_norm = batch["Δ"].to(device)
        smiles = batch["smiles"].to(device)

        # -------- SMILES embedding (frozen) --------
        with torch.no_grad():
            mol_emb = encoder(smiles)

        # -------- diffusion step --------
        t = diffusion.sample_timesteps(G_pre.size(0)).to(device)
        x_t, noise_true = diffusion.add_noise(Δ_true_norm, t)

        noise_pred = model(x_t, G_pre, mol_emb, t)
        loss_noise = F.mse_loss(noise_pred, noise_true)

        # CHANGE: Allow gradients to propagate
        alpha_bar = diffusion.alpha_bars[t].unsqueeze(1)

        Δ_pred_norm = (x_t - torch.sqrt(1 - alpha_bar) * noise_pred) / torch.sqrt(alpha_bar)

        G_post_pred = G_pre + (Δ_pred_norm * Δ_std_device + Δ_mean_device)
        G_post_true = G_pre + (Δ_true_norm * Δ_std_device + Δ_mean_device)

        G_post_pred = F.relu(G_post_pred)

        # -------- correlation & scale loss (WITH grad) --------
        loss_corr = 1 - pearson_corr(G_post_pred, G_post_true)
        loss_scale = F.mse_loss(
            G_post_pred.mean(dim=1),
            G_post_true.mean(dim=1)
        )

        loss_total = (
            λ_noise * loss_noise +
            λ_corr  * loss_corr +
            λ_scale * loss_scale
        )

        optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss_epoch += loss_total.item()
        noise_epoch += loss_noise.item()
        scale_epoch += loss_scale.item()
        corr_epoch  += loss_corr.item()

    avg_loss  = total_loss_epoch / len(loader)
    avg_noise = noise_epoch / len(loader)
    avg_scale = scale_epoch / len(loader)
    avg_corr  = corr_epoch / len(loader)

    loss_total_history.append(avg_loss)
    loss_noise_history.append(avg_noise)
    loss_scale_history.append(avg_scale)
    loss_corr_history.append(avg_corr)

    print(
        f"[Epoch {epoch+1}] "
        f"Loss={avg_loss:.4f} | "
        f"Noise={avg_noise:.4f} | "
        f"Scale={avg_scale:.4f} | "
        f"1-Pearson={avg_corr:.4f}"
    )

# ---------------- Save ----------------
os.makedirs("checkpoints", exist_ok=True)
torch.save({
    "model_state": model.state_dict(),
    "gene_cols": gene_cols,
    "Δ_mean": delta_norm["Δ_mean"].cpu(),
    "Δ_std": delta_norm["Δ_std"].cpu(),
    "diffusion_config": {
        "T": diffusion.T,
        "beta_start": 1e-4,
        "beta_end": 0.01
    },
    "model_config": {
        "gene_dim": len(gene_cols),
        "mol_dim": 2048,
        "hidden": 1024
    }
}, "checkpoints/final_model.pt")

np.save("checkpoints/loss_total.npy", np.array(loss_total_history))
np.save("checkpoints/loss_noise.npy", np.array(loss_noise_history))
np.save("checkpoints/loss_scale.npy", np.array(loss_scale_history))
np.save("checkpoints/loss_corr.npy", np.array(loss_corr_history))

print("Stage 9 finished — model saved")

# Conditional Diffusion Sampling
@torch.no_grad()
def sample_diffusion(model, encoder, smiles_tensor, G_pre, diffusion, Δ_mean, Δ_std):
    """
    Conditional G_pre + multi-step diffusion sampling (full DDPM)
    """
    model.eval()
    encoder.eval()

    batch_size, gene_dim = G_pre.size()

    mol_emb = encoder(smiles_tensor.to(G_pre.device))
    x_t = torch.randn(batch_size, gene_dim, device=G_pre.device)

    for t in reversed(range(diffusion.T)):
        t_batch = torch.full((batch_size,), t, device=G_pre.device, dtype=torch.long)

        noise_pred = model(x_t, G_pre, mol_emb, t_batch)

        alpha_t = diffusion.alphas[t]
        alpha_bar_t = diffusion.alpha_bars[t]
        beta_t = diffusion.betas[t]

        x_prev = (1 / torch.sqrt(alpha_t)) * (
            x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * noise_pred
        )

        if t > 0:
            x_prev = x_prev + torch.sqrt(beta_t) * torch.randn_like(x_t)

        x_t = x_prev

    Δ_pred = x_t * Δ_std.to(G_pre.device) + Δ_mean.to(G_pre.device)
    G_post_pred = G_pre + Δ_pred
    return G_post_pred

