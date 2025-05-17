import sys
import os
import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BartTokenizer, BartForConditionalGeneration

# Add custom paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'eeg_alignment')))
from clip_baseline import EEGTextCLIP
from bart_decoder import EEGtoBART

# === Device ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# === Load EEG encoder ===
clip_model = EEGTextCLIP(eeg_dim=256, text_dim=768, embed_dim=128)
clip_model.load_state_dict(torch.load("/content/clip_baseline_model.pth", map_location=device))
clip_model.eval()

eeg_encoder = clip_model.eeg_encoder
for param in eeg_encoder.parameters():
    param.requires_grad = False
print("Loaded frozen EEG encoder.")


# === Load dataset ===
train_data = torch.load('/content/train_embeddings.pt', map_location=device)
eeg_embeddings = train_data['eeg']
text_labels = train_data['words']
print(f"Loaded {len(eeg_embeddings)} samples.")


# === Tokenizer ===
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

class EEGToTextDataset(Dataset):
    def __init__(self, eeg_embeddings, texts, tokenizer):
        self.eeg_embeddings = eeg_embeddings
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.eeg_embeddings)

    def __getitem__(self, idx):
        eeg = self.eeg_embeddings[idx]
        text = self.texts[idx]
        inputs = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=32)
        return eeg, inputs['input_ids'].squeeze(0), inputs['attention_mask'].squeeze(0)

# === Split into train/val ===
full_dataset = EEGToTextDataset(eeg_embeddings, text_labels, tokenizer)
val_ratio = 0.2
val_size = int(val_ratio * len(full_dataset))
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
print(f"Train: {train_size}, Val: {val_size}")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# === Initialize model ===
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
model = EEGtoBART(eeg_encoder, bart_model, eeg_embed_dim=128, bart_hidden_dim=bart_model.config.d_model)
model = model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# === Load checkpoint if exists ===
start_epoch = 1
checkpoint_path = "bart_decoder_checkpoint_epoch.pth"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"✅ Resumed from checkpoint: {checkpoint_path} (epoch {checkpoint['epoch']})")

# === Training Loop ===
epochs = 12
save_every = 5
train_losses = []
val_losses = []

for epoch in range(start_epoch, epochs + 1):
    model.train()
    total_loss = 0
    for eeg_batch, input_ids, attention_mask in train_loader:
        eeg_batch = eeg_batch.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        outputs = model(eeg_batch, labels=input_ids)
        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]

        loss = loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

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
        for eeg_batch, input_ids, attention_mask in val_loader:
            eeg_batch = eeg_batch.to(device)
            input_ids = input_ids.to(device)

            outputs = model(eeg_batch, labels=input_ids)
            logits = outputs.logits[:, :-1, :]
            labels = input_ids[:, 1:]

            loss = loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

    # === Save checkpoint ===
    if epoch % save_every == 0 or epoch == epochs:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch}")

# === Save final model ===
torch.save(model.state_dict(), "bart_decoder_final.pth")
print("Final model saved!")

# === Save losses ===
torch.save({'train': train_losses, 'val': val_losses}, "loss_bart_history.pt")

# === Plot ===
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.title("EEG-BART Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_bart_curve.png")
plt.close()
