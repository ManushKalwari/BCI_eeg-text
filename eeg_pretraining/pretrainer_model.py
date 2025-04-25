import torch
import torch.nn as nn
from eeg_pretraining.eeg_quantizer import EEGPatchEmbedding
from eeg_pretraining.pos_encoding import LearnablePositionalEncoding
from eeg_pretraining.masking import EEGMasker
from eeg_pretraining.transformer_encoder import EEGTransformerEncoder

class EEGPretrainer(nn.Module):
    def __init__(self, 
                 in_channels=105, 
                 embed_dim=256,
                 patch_size=25,
                 stride=10,
                 max_len=200,
                 depth=4,
                 num_heads=8,
                 mlp_ratio=4.0,
                 mask_ratio=0.15):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.in_channels = in_channels

        self.embed = EEGPatchEmbedding(in_channels, embed_dim, patch_size, stride)
        self.pos_enc = LearnablePositionalEncoding(max_len, embed_dim)
        self.masker = EEGMasker(mask_ratio)
        self.transformer = EEGTransformerEncoder(embed_dim, depth, num_heads, mlp_ratio)
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.reconstruct = nn.Linear(embed_dim, self.patch_size * in_channels)


    def forward(self, x):
        # x: [B, T, C]
        B, T, C = x.shape
        x_patched = self.embed(x)                # [B, N, D]
        x_patched = self.pos_enc(x_patched)      # positional encoding

        mask = self.masker(x_patched)            # [B, N] boolean
        x_masked = x_patched.clone()
        x_masked[mask] = self.mask_token

        encoded = self.transformer(x_masked)     # [B, N, D]
        pred_patches = self.reconstruct(encoded) # [B, N, C]
        pred = pred_patches.view(B, -1, self.patch_size, self.in_channels)

        # === Unpatching ===
        unpatched_pred = torch.zeros(B, T, C, device=x.device)
        unpatched_mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)

        patch_idx = 0
        for i in range(0, T - self.patch_size + 1, self.stride):
            unpatched_pred[:, i:i+self.patch_size] += pred[:, patch_idx]
            unpatched_mask[:, i:i+self.patch_size] = True
            patch_idx += 1

        return unpatched_pred, unpatched_mask, x
