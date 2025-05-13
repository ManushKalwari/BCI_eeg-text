
"""
eeg_text_dataset.py

Loads EEG-word pairs from .mat files and prepares them as a PyTorch Dataset.

- Used during pretraining, CLIP training, and BART decoding.
- Extracts raw EEG signals and corresponding words from MATLAB `.mat` files.
- Provides:
    1. EEGTextDataset class for accessing (word, EEG) pairs
    2. collate_fn for padding EEG sequences (for SSL training)
    3. text_collate_fn for batching words (used for text embedding with BERT)
"""

import os
import re
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from scipy.io import loadmat


class EEGTextDataset(Dataset):
    def __init__(self, data_dir):
        self.samples = []

        # Loop over all .mat files in the directory
        for filename in os.listdir(data_dir):
            if filename.endswith(".mat"):
                mat = loadmat(os.path.join(data_dir, filename), squeeze_me=True, struct_as_record=False)
                pairs = mat['pairs']  # shape: (1, N) or (N,) depending on squeeze

                if pairs.ndim == 1:  # Correct shape
                    for i, pair in enumerate(pairs):
                        try:
                            # Extract word and normalize
                            word_raw = pair[0][0] if isinstance(pair[0], np.ndarray) else pair[0]
                            word = re.sub(r"[^\w]", "", word_raw.lower())  # remove punctuation, lowercase

                            eeg = pair[1]

                            # Validate EEG shape: expected [105, T]
                            if eeg.shape[0] != 105:
                                print(f"[{filename}] Skipping index {i} with bad EEG shape {eeg.shape}")
                                continue

                            eeg = eeg.T  # Transpose to shape [T, 105]
                            self.samples.append((word, eeg))

                        except Exception as e:
                            print(f"[{filename}] Error processing index {i}: {e}")
                else:
                    print(f"[{filename}] Unexpected pairs shape: {pairs.shape}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        word, eeg = self.samples[idx]

        # If EEG is path string (rare), load it from file
        eeg = np.load(eeg) if isinstance(eeg, str) else eeg

        return {
            'word': word,
            'eeg': torch.tensor(eeg, dtype=torch.float32)
        }


# === Collate function for EEG SSL pretraining ===
# Pads variable-length EEG time sequences to batch format
def collate_fn(batch):
    eegs = [item['eeg'] for item in batch]  # already torch tensors
    eegs_padded = pad_sequence(eegs, batch_first=True)  # shape: [B, T, C]
    return {'eeg': eegs_padded}


# === Collate function for BERT-based text embedding ===
# Returns raw words for tokenization
def text_collate_fn(batch):
    words = [item['word'] for item in batch]
    return {'word': words}
