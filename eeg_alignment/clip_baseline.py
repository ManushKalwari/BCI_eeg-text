import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# === EEG Encoder ===
class EEGEncoder(nn.Module):
    def __init__(self, eeg_dim, embed_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(eeg_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
    def forward(self, eeg):
        return self.encoder(eeg)


# === Text Encoder ===
class TextEncoder(nn.Module):
    def __init__(self, text_dim, embed_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(text_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
    def forward(self, text):
        return self.encoder(text)



# === Full CLIP Model ===
class EEGTextCLIP(nn.Module):
    def __init__(self, eeg_dim, text_dim, embed_dim=128):
        super().__init__()
        self.eeg_encoder = EEGEncoder(eeg_dim, embed_dim)
        self.text_encoder = TextEncoder(text_dim, embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))  # init with 0.07

    def forward(self, eeg, text):
        eeg_embeds = self.eeg_encoder(eeg)   # [B, D]
        text_embeds = self.text_encoder(text)  # [B, D]

        eeg_embeds = F.normalize(eeg_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)

        # Clamp logit scale to prevent it from growing too large
        scale = self.logit_scale.clamp(max=np.log(100.0)).exp()
        return eeg_embeds, text_embeds, scale



# === Contrastive Loss ===
def clip_loss(logits):
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_eeg_to_text = F.cross_entropy(logits, labels)
    loss_text_to_eeg = F.cross_entropy(logits.t(), labels)
    return (loss_eeg_to_text + loss_text_to_eeg) / 2

