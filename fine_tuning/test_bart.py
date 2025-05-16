import sys
import os

# Add eeg_alignment to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'eeg_alignment')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'eeg_pretraining')))



import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BartTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from bart_decoder import EEGtoBART
from clip_baseline import EEGTextCLIP
from eeg_text_dataset import EEGTextDataset
from transformers.modeling_outputs import BaseModelOutput

# === Settings ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# === Load EEG encoder ===
clip_model = EEGTextCLIP(eeg_dim=256, text_dim=768, embed_dim=512)
clip_model.load_state_dict(torch.load('clip_baseline_model.pth', map_location=device))
clip_model.eval()
eeg_encoder = clip_model.eeg_encoder
for param in eeg_encoder.parameters():
    param.requires_grad = False
print("Loaded frozen EEG encoder.")

# === Load tokenizer and decoder ===
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
from transformers import BartForConditionalGeneration
bart_decoder = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

# === Init EEG→BART model ===
model = EEGtoBART(
    eeg_encoder=eeg_encoder,
    bart_model=bart_decoder,
    eeg_embed_dim=512,
    bart_hidden_dim=bart_decoder.config.d_model
).to(device)

# model.load_state_dict(torch.load('/content/bart_decoder_checkpoint_epoch20.pth', map_location=device)['model_state_dict'])
# model = model.to(device)
# model.eval()
# print("Model loaded and ready.")

# 2. Then load checkpoint
checkpoint = torch.load('/content/bart_decoder_checkpoint_epoch.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# === Load test EEG embeddings ===
test_data = torch.load('/content/test_embeddings.pt', map_location=device)
eeg_test_embeddings = test_data['eeg']  # [N_test, 256]
print(f"Loaded EEG test embeddings: {eeg_test_embeddings.shape}")

# === Load corresponding reference words ===
data_dir = '/content/drive/MyDrive/BCI_trainingData/'  # Adjust if needed
text_dataset = EEGTextDataset(data_dir)
ref_words = [sample['word'] for sample in text_dataset]
ref_words = ref_words[-len(eeg_test_embeddings):]  # assuming test is the last part
print(f"Using {len(ref_words)} reference words.")

# === Dataset and Dataloader ===
class EEGTestDataset(Dataset):
    def __init__(self, eeg_embeddings, texts):
        self.eeg_embeddings = eeg_embeddings
        self.texts = texts

    def __len__(self):
        return len(self.eeg_embeddings)

    def __getitem__(self, idx):
        eeg = self.eeg_embeddings[idx]
        ref_text = self.texts[idx]
        return eeg, ref_text

test_dataset = EEGTestDataset(eeg_test_embeddings, ref_words)
test_loader = DataLoader(test_dataset, batch_size=32)

# === Metrics setup ===
rouge = Rouge()
smooth_fn = SmoothingFunction().method1

bleu_scores = {1: [], 2: [], 3: [], 4: []}
rouge_scores = []

# === Evaluation ===
all_preds, all_refs = [], []

with torch.no_grad():
    for eeg_batch, ref_batch in test_loader:
        eeg_batch = eeg_batch.to(device)

        eeg_features = model.eeg_encoder(eeg_batch)
        projected = model.project(eeg_features).unsqueeze(1)

        encoder_outputs = BaseModelOutput(last_hidden_state=projected)
        generated_ids = model.bart.generate(
            encoder_outputs=encoder_outputs,
            max_length=32,
            num_beams=5,
            early_stopping=True
        )

        predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        references = ref_batch

        # 🔍 DEBUG: print a few predictions vs. references
        for i in range(min(5, len(predictions))):
            print(f"\n🔹 Reference: {references[i]}")
            print(f"🔸 Prediction: {predictions[i]}")

        all_preds.extend(predictions)
        all_refs.extend(references)

        # Compute scores
        for pred, ref in zip(predictions, references):
            ref_tokens = ref.split()
            pred_tokens = pred.split()

            for n in range(1, 5):
                score = sentence_bleu([ref_tokens], pred_tokens, weights=tuple([1 / n] * n), smoothing_function=smooth_fn)
                bleu_scores[n].append(score)

            try:
                rouge_score = rouge.get_scores(pred, ref)[0]['rouge-1']
                rouge_scores.append(rouge_score)
            except:
                # handle empty predictions
                rouge_scores.append({'p': 0.0, 'r': 0.0, 'f': 0.0})

# === Average BLEU and ROUGE ===
print("\n=== Evaluation Results ===")
for n in range(1, 5):
    avg_bleu = sum(bleu_scores[n]) / len(bleu_scores[n])
    print(f"BLEU-{n}: {avg_bleu:.4f}")

rouge_p = sum([s['p'] for s in rouge_scores]) / len(rouge_scores)
rouge_r = sum([s['r'] for s in rouge_scores]) / len(rouge_scores)
rouge_f = sum([s['f'] for s in rouge_scores]) / len(rouge_scores)

print(f"ROUGE-1 Precision: {rouge_p:.4f}")
print(f"ROUGE-1 Recall:    {rouge_r:.4f}")
print(f"ROUGE-1 F1:        {rouge_f:.4f}")
