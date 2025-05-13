"""
eeg_transformer_encoder.py

Defines a simple multi-layer Transformer encoder for EEG patch embeddings.

Used in the EEG self-supervised pretraining stage after patching and positional encoding.
"""

import torch
import torch.nn as nn


class EEGTransformerEncoder(nn.Module):
    """
    Multi-layer Transformer encoder for processing EEG patch sequences.

    Each layer is a standard nn.TransformerEncoderLayer using GELU activation.

    Args:
        embed_dim (int): Dimension of each patch embedding (d_model)
        depth (int): Number of transformer layers
        num_heads (int): Number of self-attention heads
        mlp_ratio (float): Hidden layer size = embed_dim * mlp_ratio
        dropout (float): Dropout applied in attention and feedforward layers
    """
    def __init__(self, embed_dim=256, depth=4, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=dropout,
                activation='gelu',
                batch_first=True
            )
            for _ in range(depth)
        ])

    def forward(self, x, src_key_padding_mask=None):
        """
        Args:
            x (Tensor): Input tensor of shape [batch, seq_len, embed_dim]
            src_key_padding_mask (Tensor, optional): Optional mask for padded tokens

        Returns:
            Tensor: Encoded output of the same shape [batch, seq_len, embed_dim]
        """
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return x
