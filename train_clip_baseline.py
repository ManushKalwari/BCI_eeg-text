# train_clip_baseline.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from clip_baseline import EEGTextCLIP

# === Hyperparameters ===
batch_size = 64
lr = 2e-3
epochs = 30
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

eeg_embeddings_path = "embeddings/eeg_embeddings.pt" 
text_embeddings_path = "embeddings/text_embeddings.pt" 

# === Load embeddings ===
eeg_embeddings = torch.load(eeg_embeddings_path)  # shape [N, eeg_dim]
text_embeddings = torch.load(text_embeddings_path)  # shape [N, text_dim]

text_embeddings = text_embeddings['embeddings']

print(f"EEG Embeddings Shape: {eeg_embeddings.shape}")
print(f"Text Embeddings Shape: {text_embeddings.shape}")

assert len(eeg_embeddings) == len(text_embeddings), "Mismatch in EEG and Text samples!"

# === Dataset ===
class EmbeddingDataset(Dataset):
    def __init__(self, eeg_embeds, text_embeds):
        self.eeg = eeg_embeds
        self.text = text_embeds

    def __len__(self):
        return len(self.eeg)

    def __getitem__(self, idx):
        return {
            'eeg': self.eeg[idx],
            'text': self.text[idx]
        }

dataset = EmbeddingDataset(eeg_embeddings, text_embeddings)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# === Model ===
eeg_dim = eeg_embeddings.shape[1]
text_dim = text_embeddings.shape[1]
embed_dim = 512  # common latent dimension

model = EEGTextCLIP(eeg_dim, text_dim, embed_dim).to(device)

# === Optimizer, Loss ===
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# === Training ===
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in dataloader:
        eeg = batch['eeg'].to(device)    # [B, eeg_dim]
        text = batch['text'].to(device)  # [B, text_dim]

        optimizer.zero_grad()
        eeg_proj, text_proj = model(eeg, text)  # [B, embed_dim] each

        # === Compute logits ===
        logits_per_eeg = (eeg_proj @ text_proj.t())  # [B, B]
        logits_per_text = logits_per_eeg.t()         # [B, B]

        labels = torch.arange(len(eeg)).to(device)  # [0, 1, 2, ..., B-1]

        loss_eeg = criterion(logits_per_eeg, labels)
        loss_text = criterion(logits_per_text, labels)
        loss = (loss_eeg + loss_text) / 2

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# === Save the model ===
torch.save(model.state_dict(), "clip_baseline_model.pth")
print("Training done and model saved!")
