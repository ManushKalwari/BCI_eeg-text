import torch
import torch.nn as nn

class EEGMasker:

    #randomly select ~15% of EEG patches and tell the model to reconstruct them.

    def __init__(self, mask_ratio=0.15):
        self.mask_ratio = mask_ratio

    def __call__(self, x):
        # x: [batch, seq_len, embed_dim]
        B, L, D = x.size()
        num_mask = int(self.mask_ratio * L)

        mask = torch.zeros(B, L, dtype=torch.bool, device=x.device)
        for i in range(B):
            rand_indices = torch.randperm(L)[:num_mask]
            mask[i, rand_indices] = True

        return mask