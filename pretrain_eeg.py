import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from eeg_pretraining.pretrainer_model import EEGPretrainer
from eeg_text_dataset import EEGTextDataset, collate_fn   


# Get absolute path to your project root (one level above eeg_pretraining)
project_root = os.path.abspath(os.path.join(os.getcwd()))
if project_root not in sys.path:
    sys.path.append(project_root)



# === Config ===
batch_size = 16
epochs = 20
lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Data ===
dataset = EEGTextDataset("BCI_trainingData/")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# === Model ===
model = EEGPretrainer().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()



for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in dataloader:
        eeg = batch['eeg'].to(device)  # [B, T, C]

        pred, mask, target = model(eeg)  # unpatched prediction and mask
        loss = criterion(pred[mask], target[mask])  # MSE only on valid positions

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")




torch.save(model.state_dict(), "eeg_pretrained.pth")