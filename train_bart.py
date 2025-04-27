import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import BartTokenizer
import os
from transformers import BartForConditionalGeneration

bart_decoder = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

# === Settings ===
epochs = 20
batch_size = 32
learning_rate = 1e-4
save_every = 5  # save checkpoint every 5 epochs

# === Device ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# === Load EEG encoder from CLIP model ===
from clip_baseline import EEGTextCLIP  # your CLIP model

clip_checkpoint_path = "clip_baseline_model.pth"  # your CLIP trained checkpoint

clip_model = EEGTextCLIP(eeg_dim=256, text_dim=768, embed_dim=512)
clip_model.load_state_dict(torch.load(clip_checkpoint_path, map_location=device))
clip_model.eval()

eeg_encoder = clip_model.eeg_encoder  # extract
for param in eeg_encoder.parameters():
    param.requires_grad = False  # freeze
print("Loaded frozen EEG encoder.")

# === Load your EEGTextBART model ===
from bart_decoder import EEGtoBART  # make sure your model code is ready
from eeg_text_dataset import EEGTextDataset  # to get raw words

# === Load train embeddings ===
train_data = torch.load('embeddings/train_embeddings.pt', map_location=device)
eeg_train_embeddings = train_data['eeg']  # shape [N_train, 256]
print(f"Loaded EEG train embeddings: {eeg_train_embeddings.shape}")

# === Load words from EEGTextDataset ===
data_dir = '/content/drive/MyDrive/BCI_trainingData/'  # adjust if needed
text_dataset = EEGTextDataset(data_dir)

all_words = [sample['word'] for sample in text_dataset]
print(f"Loaded {len(all_words)} words from text dataset.")

# === Match words to EEG embeddings ===
train_words = all_words[:len(eeg_train_embeddings)]
print(f"Using {len(train_words)} words aligned with EEG embeddings.")

# === Load tokenizer ===
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

# === Define Dataset ===
class EEGToTextDataset(Dataset):
  # this is difefrent fom the one in eeg_text_dataset, that uses raw eeg, here we use embeddings
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
        input_ids = inputs['input_ids'].squeeze(0)  # (seq_len)
        attention_mask = inputs['attention_mask'].squeeze(0)

        return eeg, input_ids, attention_mask

# === Create Dataset and Loader ===
train_dataset = EEGToTextDataset(eeg_train_embeddings, train_words, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# === Load Model ===
model = EEGtoBART(
    eeg_encoder=eeg_encoder,
    bart_model=bart_decoder,
    eeg_embed_dim=512,
    bart_hidden_dim=bart_decoder.config.d_model
)
model = model.to(device)

# === Optimizer ===
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# === Loss Function ===
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# === Training Loop ===
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0

    for eeg_batch, input_ids, attention_mask in train_loader:
        eeg_batch = eeg_batch.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        outputs = model(eeg_batch, labels=input_ids)

        logits = outputs.logits[:, :-1, :].contiguous()  # predict next token
        labels = input_ids[:, 1:].contiguous()  # shifted target

        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")

    # === Save checkpoint ===
    if epoch % save_every == 0 or epoch == epochs:
        save_path = f"bart_decoder_checkpoint_epoch{epoch}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, save_path)
        print(f"Checkpoint saved at {save_path}!")

print("Training complete ✅")
