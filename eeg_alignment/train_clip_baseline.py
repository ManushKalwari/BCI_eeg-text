"""
train_clip_baseline.py (with validation)

Trains a CLIP-style model to align EEG and Text embeddings with contrastive loss.
Adds validation split, loss tracking, and plots.
"""

import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim
import matplotlib.pyplot as plt
from clip_baseline import EEGTextCLIP

# === Hyperparameters ===
batch_size = 256
lr = 1e-6
epochs = 500
val_ratio = 0.2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Load full train embeddings ===
data = torch.load('/content/train_embeddings.pt')
eeg_all = data['eeg']
text_all = data['text']
assert len(eeg_all) == len(text_all)

print(f"Loaded {len(eeg_all)} training samples.")

# === Dataset ===
class EmbeddingDataset(Dataset):
    def __init__(self, eeg, text):
        self.eeg = eeg
        self.text = text

    def __len__(self):
        return len(self.eeg)

    def __getitem__(self, idx):
        return {'eeg': self.eeg[idx], 'text': self.text[idx]}

# Create dataset and split
full_dataset = EmbeddingDataset(eeg_all, text_all)
val_size = int(len(full_dataset) * val_ratio)
train_size = len(full_dataset) - val_size
train_set, val_set = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# === Model ===
eeg_dim = eeg_all.shape[1]
text_dim = text_all.shape[1]
embed_dim = 512

model = EEGTextCLIP(eeg_dim, text_dim, embed_dim).to(device)
optimizer = optim.AdamW(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# === Checkpoint ===
checkpoint_path = "clip_baseline_checkpoint.pth"
loss_path = "loss_clip_history.pt"
start_epoch = 0
train_losses = []
val_losses = []

# === Resume if available ===
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resumed from epoch {start_epoch}")

# === Load previous loss history if any ===
if os.path.exists(loss_path):
    saved = torch.load(loss_path)
    train_losses = saved['train']
    val_losses = saved['val']
    print(f"Loaded previous loss history ({len(train_losses)} epochs)")

# === Training ===
for epoch in range(start_epoch, epochs):
    model.train()
    total_train_loss = 0

    for batch in train_loader:
        eeg = batch['eeg'].to(device)
        text = batch['text'].to(device)

        optimizer.zero_grad()
        
        # === Get projections and scale ===
        eeg_proj, text_proj, scale = model(eeg, text)

        # === Contrastive logits ===
        logits_eeg = (eeg_proj @ text_proj.t()) * scale
        logits_text = logits_eeg.t()

        labels = torch.arange(len(eeg)).to(device)
        loss = (criterion(logits_eeg, labels) + criterion(logits_text, labels)) / 2

        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()

        

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # === Validation ===
    model.eval()
    total_val_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            eeg = batch['eeg'].to(device)
            text = batch['text'].to(device)

            eeg_proj, text_proj, scale = model(eeg, text)

            logits_eeg = (eeg_proj @ text_proj.t()) * scale
            logits_text = logits_eeg.t()

            labels = torch.arange(len(eeg)).to(device)
            loss = (criterion(logits_eeg, labels) + criterion(logits_text, labels)) / 2
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # === Save Checkpoint ===
    if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch+1}")

# === Save Final Model ===
torch.save(model.state_dict(), "clip_baseline_model.pth")
print("Training complete and model saved.")

# === Save Loss History ===
torch.save({'train': train_losses, 'val': val_losses}, loss_path)
print(f"Saved loss history to {loss_path}")

# === Plot ===
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Train Loss', linewidth=2)
plt.plot(val_losses, label='Val Loss', linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Contrastive Loss")
plt.title("CLIP Training Loss (Train vs. Val)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("loss_clip_curve.png")
plt.close()
