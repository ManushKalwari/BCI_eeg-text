# save_text_embeddings.py

import os
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from eeg_text_dataset import EEGTextDataset, text_collate_fn
from tqdm import tqdm

# ====== SETTINGS ======
data_dir = "/content/drive/MyDrive/BCI_trainingData/"
save_path = "/content/drive/MyDrive/BCI_trainingData/text_embeddings.pt"
batch_size = 64  # You can increase if memory allows
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== LOAD DATASET ======
dataset = EEGTextDataset(data_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=text_collate_fn)


# ====== LOAD BERT MODEL ======
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
bert_model.eval()

# ====== STORAGE ======
all_embeddings = []
all_words = []

# ====== PROCESS ======
with torch.no_grad():
    for batch in tqdm(dataloader):
        words = batch['word']  # list of words
        
        # Tokenize
        encoded = tokenizer(words, padding=True, truncation=True, return_tensors="pt")
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        # Pass through BERT
        outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token embedding
        
        all_embeddings.append(embeddings.cpu())
        all_words.extend(words)

# ====== SAVE ======
all_embeddings = torch.cat(all_embeddings, dim=0)  # shape [N, 768]
save_dict = {
    'embeddings': all_embeddings,  # tensor
    'words': all_words             # list of strings
}
torch.save(save_dict, save_path)

print(f"Saved text embeddings to {save_path}")
