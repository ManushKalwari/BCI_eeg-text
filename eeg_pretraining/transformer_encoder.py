import torch
import torch.nn as nn

class EEGTransformerEncoder(nn.Module):
    def __init__(self, embed_dim=256, depth=4, num_heads=8, mlp_ratio=4.0, dropout=0.1):
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
        # x: [batch, seq_len, embed_dim]
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return x
