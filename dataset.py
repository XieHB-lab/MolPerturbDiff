#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random
import pickle
import numpy as np
from tqdm import tqdm

# Stable SMILES Tokenizer

PAD_IDX  = 0
MASK_IDX = 1
UNK_IDX  = 2

def build_vocab(smiles_list):
    chars = sorted(set("".join(smiles_list)))
    vocab = {c: i+3 for i,c in enumerate(chars)}
    return vocab

def smiles_to_tensor(smiles, vocab, max_len=128):
    idxs = [vocab.get(c, UNK_IDX) for c in smiles[:max_len]]
    if len(idxs) < max_len:
        idxs += [PAD_IDX]*(max_len-len(idxs))
    return torch.LongTensor(idxs)

# Encoder Contrastive Pretrain dataset

class SmilesPretrainDataset(Dataset):
    def __init__(self, smiles_list, vocab, max_len=128, mask_prob=0.15):
        self.smiles = smiles_list
        self.vocab = vocab
        self.max_len = max_len
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        s = self.smiles[idx]
        x = smiles_to_tensor(s, self.vocab, self.max_len)
        x_mask = x.clone()

        nonpad = (x != PAD_IDX).nonzero(as_tuple=True)[0]
        for i in nonpad:
            if random.random() < self.mask_prob:
                x_mask[i] = MASK_IDX
        return x, x_mask 改成这样？

# Build Δ using Pre/Post matching (Batch/Replicate DMSO)
def build_delta_pairs(
    expr_df,
    train_s2,
    gene_cols,
    fold_idx,
    cache_dir="."
):
    pairs_file = os.path.join(cache_dir, f"pairs_fold{fold_idx}.pkl")
    delta_norm_file = os.path.join(cache_dir, f"delta_norm_fold{fold_idx}.pt")

    # ---- Load cache if exists ----
    if os.path.exists(pairs_file) and os.path.exists(delta_norm_file):
        print("[Stage6] Found cached pairs and delta_norm. Loading...")
        with open(pairs_file, "rb") as f:
            pairs = pickle.load(f)
        delta_norm = torch.load(delta_norm_file)
        return pairs, delta_norm

    print("[Stage6] Building Δ pairs from Pre/Post matching...")
    pairs = []

    for mid in tqdm(train_s2, desc="[Stage6] Molecules"):

        post_df = expr_df[
            (expr_df["Molecule_id"] == mid) &
            (expr_df["Treatment_status"] == "Post")
        ]

        if len(post_df) == 0:
            continue

        for _, post_row in post_df.iterrows():

            batch_id = post_row["Batch_id"]
            rep_id   = post_row["Replicate_id"]

            pre_df = expr_df[
                (expr_df["Molecule_id"] == "DMSO") &
                (expr_df["Treatment_status"] == "Pre") &
                (expr_df["Batch_id"] == batch_id) &
                (expr_df["Replicate_id"] == rep_id)
            ]

            if len(pre_df) == 0:
                pre_df = expr_df[
                    (expr_df["Molecule_id"] == "DMSO") &
                    (expr_df["Treatment_status"] == "Pre") &
                    (expr_df["Batch_id"] == batch_id)
                ]

            if len(pre_df) == 0:
                continue

            G_pre_mean = pre_df[gene_cols].values.astype("float32").mean(axis=0)
            G_post_vals = post_row[gene_cols].values.astype("float32")

            delta = G_post_vals - G_pre_mean

            pairs.append({
                "G_pre": G_pre_mean,
                "Δ": delta,
                "Molecule_id": mid,
                "Condition_id": post_row["Condition_id"]
            })

    if len(pairs) == 0:
        raise ValueError("No Δ pairs found!")

    # ---- Compute normalization ----
    delta_all = torch.tensor(
        np.stack([p["Δ"] for p in pairs]),
        dtype=torch.float32
    )

    delta_mean = delta_all.mean(dim=0)
    delta_std  = torch.clamp(delta_all.std(dim=0), min=0.1)

    delta_norm = {
        "Δ_mean": delta_mean,
        "Δ_std": delta_std
    }

    # ---- Save cache ----
    with open(pairs_file, "wb") as f:
        pickle.dump(pairs, f)

    torch.save(delta_norm, delta_norm_file)

    print(f"[Stage6] Saved pairs to {pairs_file}")

    return pairs, delta_norm

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

