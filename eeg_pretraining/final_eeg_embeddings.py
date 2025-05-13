import torch
from torch.utils.data import DataLoader
from eeg_text_dataset import EEGTextDataset, collate_fn  
from eeg_pretraining.pretrainer_model import EEGPretrainer  # adjust import if needed
from tqdm import tqdm
import os

# Paths
data_dir = "/content/drive/MyDrive/BCI_trainingData/"
save_path = "/content/drive/MyDrive/eeg_embeddings.pt"
checkpoint_path = "checkpoints/eeg_pretrained_model.pth"

batch_size = 16

# Load dataset
dataset = EEGTextDataset("/content/drive/MyDrive/BCI_trainingData/")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


# Only need EEG for this
def eeg_collate_fn(batch):
    eegs = [item['eeg'] for item in batch]
    return torch.stack(eegs, dim=0)


# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EEGPretrainer().to(device)
model.load_state_dict(torch.load(checkpoint_path))
model.eval()

# Save all embeddings
all_embeddings = []

with torch.no_grad():
    for batch in tqdm(dataloader):
        eeg = batch['eeg'].to(device)  # <-- FIXED: extract 'eeg' first
        
        # === Get embeddings ===
        x_patched = model.embed(eeg)             # [B, N, D]
        x_patched = model.pos_enc(x_patched)      # add positional encoding
        embeddings = model.transformer(x_patched) # [B, N, D]
        
        # === Mean Pooling ===
        global_embedding = embeddings.mean(dim=1)  # [B, D]
        
        all_embeddings.append(global_embedding.cpu())

# Stack and save
all_embeddings = torch.cat(all_embeddings, dim=0)  # [Total_samples, D]
torch.save(all_embeddings, save_path)

print(f"Saved EEG embeddings at {save_path}")
