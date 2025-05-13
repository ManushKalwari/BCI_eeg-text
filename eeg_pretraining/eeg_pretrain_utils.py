"""
eeg_pretrain_utils.py

Contains utility modules used in the self-supervised EEG pretraining stage:

1. EEGPatchEmbedding – converts raw EEG time-series into overlapping patches using 1D convolutions.
2. LearnablePositionalEncoding – adds learnable position embeddings to patch sequences.
3. EEGMasker – randomly masks a subset of patches for reconstruction (used in masked prediction).
"""

import torch
import torch.nn as nn


class EEGMasker:
    """
    Randomly selects ~15% of patches in each EEG sequence to be masked during self-supervised learning.
    The model learns to reconstruct the masked patches.
    """
    def __init__(self, mask_ratio=0.15):
        self.mask_ratio = mask_ratio

    def __call__(self, x):
        """
        Args:
            x (Tensor): [batch, seq_len, embed_dim] – input patch embeddings
        Returns:
            mask (Tensor): [batch, seq_len] boolean mask (True = masked)
        """
        B, L, D = x.size()
        num_mask = int(self.mask_ratio * L)

        mask = torch.zeros(B, L, dtype=torch.bool, device=x.device)
        for i in range(B):
            rand_indices = torch.randperm(L)[:num_mask]
            mask[i, rand_indices] = True

        return mask


class LearnablePositionalEncoding(nn.Module):
    """
    Adds a learnable positional embedding to each patch embedding to encode temporal position.
    """
    def __init__(self, max_len, embed_dim):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        """
        Args:
            x (Tensor): [batch, seq_len, embed_dim]
        Returns:
            x + pos_embed: same shape, with positional context added
        """
        return x + self.pos_embed[:, :x.size(1), :]


class EEGPatchEmbedding(nn.Module):
    """
    Converts raw EEG time-series into patch embeddings using a 1D convolutional layer.

    This approximates patch extraction (similar to ViT or MAE for vision) but for time-domain EEG data.
    """
    def __init__(self, in_channels=105, embed_dim=256, patch_size=25, stride=10):
        super().__init__()
        self.proj = nn.Conv1d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): [batch, time, channels] – raw EEG
        Returns:
            x (Tensor): [batch, num_patches, embed_dim]
        """
        x = x.permute(0, 2, 1)       # [B, C, T]
        x = self.proj(x)             # [B, embed_dim, num_patches]
        x = x.permute(0, 2, 1)       # [B, num_patches, embed_dim]
        return x
