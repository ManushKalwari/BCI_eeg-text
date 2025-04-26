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
        for filename in os.listdir(data_dir):
            if filename.endswith(".mat"):
                mat = loadmat(os.path.join(data_dir, filename), squeeze_me=True, struct_as_record=False)
                pairs = mat['pairs']  # shape: (1, N) or (N,) after squeeze_me

                if pairs.ndim == 1:  # shape (N,)
                    for i, pair in enumerate(pairs):
                        try:
                            word_raw = pair[0][0] if isinstance(pair[0], np.ndarray) else pair[0]
                            word = re.sub(r"[^\w]", "", word_raw.lower())  # strip punctuation and lowercase

                            eeg = pair[1]
                            #print(f"[{filename}] Index {i}: word = '{word}', eeg shape = {eeg.shape}")
                            
                            if eeg.shape[0] != 105:
                                print(f"[{filename}] Skipping index {i} with bad EEG shape {eeg.shape}")
                                continue

                            eeg = eeg.T  # shape (98, 105)
                            self.samples.append((word, eeg))

                        except Exception as e:
                            print(f"[{filename}] Error processing index {i}: {e}")
                else:
                    print(f"[{filename}] Unexpected pairs shape: {pairs.shape}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        
        word, eeg = self.samples[idx]
        #print("DEBUG:", type(eeg), eeg)  # check this

        eeg = np.load(eeg) if isinstance(eeg, str) else eeg
        return {
            'word': word,
            'eeg': torch.tensor(eeg, dtype=torch.float32)
        }




# eeg sequence for each word isn't uniform so we need to pad to max length
# also here we just select EEG, not word, since we arw doing SSL

def collate_fn(batch):
    eegs = [item['eeg'] for item in batch]  # already tensors
    eegs_padded = pad_sequence(eegs, batch_first=True)  # (B, T, C)
    return {'eeg': eegs_padded}