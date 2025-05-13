
"""
text_embeddings.py

Generates BERT-based embeddings for each word in the EEG-text dataset.

- Loads EEGTextDataset (.mat files)
- Extracts raw words and tokenizes using BERT tokenizer
- Uses BERT [CLS] token embeddings as fixed text representations
- Saves result as a .pt file containing:
    {
        'embeddings': Tensor of shape [N, 768],
        'words': List[str]
    }

Usage:
    python text_embeddings.py --data_dir /path/to/mat/files --save_path ./text_embeddings.pt --batch_size 64 --device cuda
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from eeg_text_dataset import EEGTextDataset, text_collate_fn
from tqdm import tqdm

def main(args):
    # === Device ===
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # === Load Dataset ===
    dataset = EEGTextDataset(args.data_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=text_collate_fn)

    # === Load BERT ===
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
    bert_model.eval()

    all_embeddings = []
    all_words = []

    # === Generate Embeddings ===
    with torch.no_grad():
        for batch in tqdm(dataloader):
            words = batch['word']
            encoded = tokenizer(words, padding=True, truncation=True, return_tensors="pt")
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)

            outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token

            all_embeddings.append(embeddings.cpu())
            all_words.extend(words)

    all_embeddings = torch.cat(all_embeddings, dim=0)

    # === Save ===
    save_dict = {
        'embeddings': all_embeddings,
        'words': all_words
    }
    torch.save(save_dict, args.save_path)
    print(f"Saved text embeddings to {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate BERT-based word embeddings for EEG dataset")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to directory containing .mat EEG files")
    parser.add_argument('--save_path', type=str, default='text_embeddings.pt', help="Path to save the output .pt file")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for BERT inference")
    parser.add_argument('--device', type=str, default='cuda', help="Device to use: 'cuda' or 'cpu'")

    args = parser.parse_args()
    main(args)
