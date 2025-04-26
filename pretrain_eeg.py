import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from lion_pytorch import Lion

from eeg_pretraining.pretrainer_model import EEGPretrainer
from eeg_text_dataset import EEGTextDataset, collate_fn   

# === Add checkpoint config ===
checkpoint_path = "checkpoints/eeg_pretrained.pth"  # <-- NEW
start_epoch = 0  # <-- NEW

# === Setup ===
project_root = os.path.abspath(os.path.join(os.getcwd()))
if project_root not in sys.path:
    sys.path.append(project_root)

batch_size = 16
epochs = 160
lr = 5e-7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Data ===
dataset = EEGTextDataset("/content/drive/MyDrive/BCI_trainingData/")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# === Model ===
model = EEGPretrainer().to(device)
# used it for first 200 epochs
optimizer = optim.AdamW(model.parameters(), lr=lr) 
#optimizer = Lion(model.parameters(), lr=lr, weight_decay=0.01)
criterion = nn.MSELoss()

# === Load checkpoint if exists ===
if os.path.exists(checkpoint_path):  # <-- NEW
    print("Loading checkpoint...")  # <-- NEW
    checkpoint = torch.load(checkpoint_path)  # <-- NEW
    model.load_state_dict(checkpoint['model_state_dict'])  # <-- NEW
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # <-- NEW
    start_epoch = checkpoint['epoch'] + 1  # <-- NEW
    print(f"Resuming from epoch {start_epoch}")  # <-- NEW

# === Training ===
for epoch in range(start_epoch, epochs):  # <-- MODIFIED
    model.train()
    total_loss = 0
    for batch in dataloader:
        eeg = batch['eeg'].to(device)

        pred, mask, target = model(eeg)
        loss = criterion(pred[mask], target[mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    # === Save checkpoint ===
    os.makedirs("checkpoints", exist_ok=True)  # <-- NEW
    torch.save({  # <-- NEW
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, checkpoint_path)

# === Optional: Save final model separately ===
torch.save(model.state_dict(), "checkpoints/eeg_pretrained_model.pth")
