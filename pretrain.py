#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm
device = "cuda" if torch.cuda.is_available() else "cpu"

# Encoder Contrastive Pretrain

encoder_ckpt_path = "encoder_pretrain_contrastive.pt"

if os.path.exists(encoder_ckpt_path):
    print("[Stage5] Found pretrained encoder. Loading...")
    ckpt = torch.load(encoder_ckpt_path, map_location=device)
    encoder = SMILESEncoder(ckpt["vocab_size"]).to(device)
    encoder.load_state_dict(ckpt["encoder_state"])
    vocab = ckpt["vocab"]
else:
    print("[Stage5] No pretrained encoder found. Training from scratch...")

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
            return x, x_mask

    dataset = SmilesPretrainDataset(all_smiles, vocab)
    loader  = DataLoader(dataset, batch_size=32, shuffle=True)

    encoder = SMILESEncoder(VOCAB_SIZE).to(device)
    opt = torch.optim.Adam(encoder.parameters(), lr=1e-4)
    temperature = 0.1

    for epoch in range(5):
        encoder.train()
        loop = tqdm(loader, desc=f"Stage5 Epoch {epoch+1}/5", ncols=100)
        for x, x_mask in loop:
            x, x_mask = x.to(device), x_mask.to(device)
            B = x.size(0)

            z1 = encoder(x)
            z2 = encoder(x_mask)

            z = torch.cat([z1,z2], 0)
            sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1)/temperature

            mask = torch.eye(2*B, device=device).bool()
            sim.masked_fill_(mask, -9e15)

            labels = torch.arange(B, device=device)
            loss = F.cross_entropy(sim[:B, B:], labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            loop.set_postfix(loss=loss.item())

    # ---- save encoder ----
    torch.save({
        "encoder_state": encoder.state_dict(),
        "vocab": vocab,
        "vocab_size": VOCAB_SIZE
    }, encoder_ckpt_path)
    print(f"[Stage5] Pretrained encoder saved to {encoder_ckpt_path}")

