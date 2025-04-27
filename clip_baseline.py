import torch
import torch.nn as nn
import torch.nn.functional as F

# === Simplified MLP Encoder for EEG ===
class EEGEncoder(nn.Module):
    def __init__(self, eeg_dim, embed_dim):
        super().__init__()
        self.fc = nn.Linear(eeg_dim, embed_dim)

    def forward(self, eeg):
        return self.fc(eeg)

# === Simplified MLP Encoder for Text ===
class TextEncoder(nn.Module):
    def __init__(self, text_dim, embed_dim):
        super().__init__()
        self.fc = nn.Linear(text_dim, embed_dim)

    def forward(self, text):
        return self.fc(text)

# === Full CLIP Model ===
class EEGTextCLIP(nn.Module):
    def __init__(self, eeg_dim, text_dim, embed_dim=512):
        super().__init__()
        self.eeg_encoder = EEGEncoder(eeg_dim, embed_dim)
        self.text_encoder = TextEncoder(text_dim, embed_dim)

    def forward(self, eeg, text):
        eeg_embeds = self.eeg_encoder(eeg)   # (B, D)
        text_embeds = self.text_encoder(text)  # (B, D)

        eeg_embeds = F.normalize(eeg_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)

        return eeg_embeds, text_embeds

# === Contrastive Loss Function ===
def clip_loss(logits):
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_eeg_to_text = F.cross_entropy(logits, labels)
    loss_text_to_eeg = F.cross_entropy(logits.t(), labels)
    return (loss_eeg_to_text + loss_text_to_eeg) / 2
