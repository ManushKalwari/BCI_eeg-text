from transformers import BartTokenizer, BartForConditionalGeneration
import torch
import torch.nn as nn
from clip_baseline import EEGTextCLIP

# === Load CLIP model checkpoint ===
checkpoint_path = "clip_baseline_model.pth"

# === Hyperparameters ===
eeg_dim = 256
text_dim = 768
embed_dim = 512

# Load the trained CLIP model
clip_model = EEGTextCLIP(eeg_dim=eeg_dim, text_dim=text_dim, embed_dim=embed_dim)
clip_model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
clip_model.eval()

# Extract only the EEG encoder
eeg_encoder = clip_model.eeg_encoder

# === Load pretrained BART model and tokenizer ===
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
bart_decoder = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

# === Define EEG → BART model ===
class EEGtoBART(nn.Module):
    def __init__(self, eeg_encoder, bart_model, eeg_embed_dim, bart_hidden_dim):
        super().__init__()
        self.eeg_encoder = eeg_encoder  # frozen EEG encoder
        self.bart = bart_model          # BART model
        self.project = nn.Linear(eeg_embed_dim, bart_hidden_dim)  # project EEG to BART input

    def forward(self, eeg, labels=None):
        with torch.no_grad():
            eeg_features = self.eeg_encoder(eeg)  # [B, embed_dim]

        eeg_features = self.project(eeg_features).unsqueeze(1)  # [B, 1, hidden_dim]

        encoder_outputs = (eeg_features,)  # wrapped as tuple!

        outputs = self.bart(encoder_outputs=encoder_outputs, labels=labels)
        return outputs



# === Setup Full Model ===
bart_hidden_dim = bart_decoder.config.d_model  # usually 768

model = EEGtoBART(
    eeg_encoder=eeg_encoder,
    bart_model=bart_decoder,
    eeg_embed_dim=embed_dim,
    bart_hidden_dim=bart_hidden_dim
)

print("Model ready: EEG Encoder → Projection → BART Decoder!")
