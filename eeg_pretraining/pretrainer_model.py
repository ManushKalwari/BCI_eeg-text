"""
pretrainer_model.py

Defines EEGPretrainer – a self-supervised masked autoencoder for EEG signals.

Workflow:
    1. Converts EEG signals into patches using 1D convolution
    2. Adds learnable positional encodings
    3. Masks a subset of patches randomly (like MAE or BERT)
    4. Encodes with Transformer encoder
    5. Predicts/reconstructs original masked EEG patches
    6. Reassembles prediction into original shape (unpatching)

Loss: MSE between predicted and original EEG at masked locations
"""

import torch
import torch.nn as nn
from eeg_pretrain_utils import EEGPatchEmbedding, EEGMasker, PositionalEncoding
from eeg_transformer_encoder import EEGTransformerEncoder


class EEGPretrainer(nn.Module):
    """
    Masked self-supervised EEG encoder using patching, masking, and Transformer encoding.
    
    Args:
        in_channels (int): Number of EEG channels (default: 105)
        embed_dim (int): Patch embedding dimension
        patch_size (int): Size of each patch (along time axis)
        stride (int): Patch stride (controls overlap)
        max_len (int): Max number of patches (used for positional encoding)
        depth (int): Transformer depth
        num_heads (int): Number of self-attention heads
        mlp_ratio (float): Feedforward expansion ratio
        mask_ratio (float): Percentage of patches to mask during training
    """
    def __init__(self, 
                 in_channels=105, 
                 embed_dim=256,
                 patch_size=25,
                 stride=10,
                 max_len=200,
                 depth=2,
                 num_heads=4,
                 mlp_ratio=4.0,
                 mask_ratio=0.15):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.in_channels = in_channels

        self.embed = EEGPatchEmbedding(in_channels, embed_dim, patch_size, stride)
        self.pos_enc = PositionalEncoding(max_len, embed_dim)
        self.masker = EEGMasker(mask_ratio)
        self.transformer = EEGTransformerEncoder(embed_dim, depth, num_heads, mlp_ratio)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.reconstruct = nn.Linear(embed_dim, patch_size * in_channels)

    def forward(self, x):
        """
        Args:
            x (Tensor): Raw EEG input of shape [B, T, C]
        Returns:
            unpatched_pred (Tensor): Reconstructed EEG [B, T, C]
            unpatched_mask (Tensor): Boolean mask indicating reconstructed areas [B, T]
            x (Tensor): Original input (for computing loss)
        """
        B, T, C = x.shape

        # === Patch + Positional Encoding ===
        x_patched = self.embed(x)                # [B, N, D]
        x_patched = self.pos_enc(x_patched)      # Add positional info

        # === Masking ===
        mask = self.masker(x_patched)            # [B, N] boolean mask
        x_masked = x_patched.clone()
        x_masked[mask] = self.mask_token

        # === Transformer Encoder ===
        encoded = self.transformer(x_masked)     # [B, N, D]

        # === Patch Reconstruction ===
        pred_patches = self.reconstruct(encoded)             # [B, N, patch_size * in_channels]
        pred = pred_patches.view(B, -1, self.patch_size, C)  # [B, N, patch_size, C]

        # === Unpatching ===
        unpatched_pred = torch.zeros(B, T, C, device=x.device)
        unpatched_mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)

        patch_idx = 0
        for i in range(0, T - self.patch_size + 1, self.stride):
            unpatched_pred[:, i:i + self.patch_size] += pred[:, patch_idx]
            unpatched_mask[:, i:i + self.patch_size] = True
            patch_idx += 1

        return unpatched_pred, unpatched_mask, x
