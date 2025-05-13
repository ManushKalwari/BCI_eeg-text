"""
final_eeg_embeddings.py

Generates fixed EEG embeddings from a pretrained EEGPretrainer model.

- Loads preprocessed EEG signals from .mat files using EEGTextDataset
- Feeds each EEG sample through patching, positional encoding, and transformer
- Applies mean pooling to get a global embedding per sample
- Saves all embeddings as a single tensor (.pt)

Output:
    eeg_embeddings.pt — Tensor of shape [N, D]
"""

import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

# === Imports ===
from eeg_text_dataset import EEGTextDataset, collate_fn
from pretrainer_model import EEGPretrainer  # Assumes same directory or proper relative import

# === Config ===
data_dir = "/content/drive/MyDrive/BCI_trainingData/"
save_path = "/content/drive/MyDrive/eeg_embeddings.pt"
checkpoint_path = "checkpoints/eeg_pretrained_model.pth"
batch_size = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Dataset ===
dataset = EEGTextDataset(data_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# === Load Pretrained EEG Model ===
model = EEGPretrainer().to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()
print("Loaded EEGPretrainer from checkpoint!")

# === Inference: Extract Embeddings ===
all_embeddings = []

with torch.no_grad():
    for batch in tqdm(dataloader):
        eeg = batch['eeg'].to(device)  # [B, T, C]

        # Pass through embedder and transformer
        x_patched = model.embed(eeg)             # [B, N, D]
        x_patched = model.pos_enc(x_patched)     # Positional encoding
        embeddings = model.transformer(x_patched)  # [B, N, D]

        global_embedding = embeddings.mean(dim=1)  # Mean-pool over patches → [B, D]
        all_embeddings.append(global_embedding.cpu())

# === Save to Disk ===
all_embeddings = torch.cat(all_embeddings, dim=0)  # [N, D]
torch.save(all_embeddings, save_path)
print(f"Saved EEG embeddings at {save_path}")
