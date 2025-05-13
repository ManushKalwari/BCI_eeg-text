"""
pretrain_eeg.py

Trains the EEGPretrainer model using masked autoencoding on raw EEG data, with train/validation split.

Workflow:
- Loads EEG-word pairs from .mat files (using EEGTextDataset)
- Splits into training and validation sets
- Applies masking and reconstruction loss (MSE) over selected patches
- Supports resuming from checkpoint
- Tracks and visualizes both training and validation loss

Output:
- checkpoints/eeg_pretrained.pth (checkpoint with optimizer state)
- checkpoints/eeg_pretrained_model.pth (final model only)
- checkpoints/loss_history.pt (train + val losses)
- checkpoints/loss_curve.png (visual loss plot)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#from lion_pytorch import Lion

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from data_preprocessing.eeg_text_dataset import EEGTextDataset, collate_fn
from pretrainer_model import EEGPretrainer

# === Paths & Hyperparameters ===
checkpoint_path = "checkpoints/eeg_pretrained.pth"
final_model_path = "checkpoints/eeg_pretrained_model.pth"
loss_history_path = "checkpoints/loss_history.pt"
loss_curve_path = "checkpoints/loss_curve.png"

batch_size = 16
epochs = 3
lr = 5e-7
start_epoch = 0

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Load Dataset and Split ===
dataset = EEGTextDataset("/content/drive/MyDrive/BCI_trainingData/")
train_indices, val_indices = train_test_split(list(range(len(dataset))), test_size=0.1, random_state=42)
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# === Model ===
model = EEGPretrainer().to(device)

# === Optimizer & Loss ===
optimizer = optim.AdamW(model.parameters(), lr=lr)
# optimizer = Lion(model.parameters(), lr=lr, weight_decay=0.01)
criterion = nn.MSELoss()

# === Resume Checkpoint ===
train_losses, val_losses = [], []

if os.path.exists(checkpoint_path):
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resuming from epoch {start_epoch}")

    # Load previous loss history if available
    if os.path.exists(loss_history_path):
        loss_history = torch.load(loss_history_path)
        train_losses = loss_history.get('train', [])
        val_losses = loss_history.get('val', [])

# === Training Loop ===
for epoch in range(start_epoch, epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        eeg = batch['eeg'].to(device)
        pred, mask, target = model(eeg)
        loss = criterion(pred[mask], target[mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # === Validation ===
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            eeg = batch['eeg'].to(device)
            pred, mask, target = model(eeg)
            loss = criterion(pred[mask], target[mask])
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # === Save Checkpoint ===
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_train_loss,
    }, checkpoint_path)

    # === Save loss history ===
    torch.save({'train': train_losses, 'val': val_losses}, loss_history_path)

# === Save Final Model ===
torch.save(model.state_dict(), final_model_path)
print(f"Final model saved to {final_model_path}")

# === Plot Loss Curve ===
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss", linewidth=2)
plt.plot(val_losses, label="Val Loss", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("EEG Pretraining Loss Curve")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(loss_curve_path)
plt.close()
print(f"Saved loss curve to {loss_curve_path}")
