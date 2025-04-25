import torch
import torch.nn as nn


class EEGPatchEmbedding(nn.Module):

    # break raw EEG into short "patches" of useful signals, we used CNN here
    #
    
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
        # x: [batch, time, channels]
        x = x.permute(0, 2, 1)  # -> [batch, channels, time]
        x = self.proj(x)        # -> [batch, embed_dim, num_patches]
        x = x.permute(0, 2, 1)  # -> [batch, num_patches, embed_dim]
        return x


# if we want more layers of CNN use this

# class EEGPatchEmbedding(nn.Module):
#     def __init__(self, in_channels=105, embed_dim=256):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv1d(in_channels, 128, kernel_size=7, stride=2, padding=3),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Conv1d(128, embed_dim, kernel_size=5, stride=2, padding=2),
#             nn.BatchNorm1d(embed_dim),
#             nn.ReLU()
#         )

#     def forward(self, x):
#         # x: [B, T, C] → [B, C, T]
#         x = x.permute(0, 2, 1)
#         x = self.encoder(x)  # [B, D, N_patches]
#         x = x.permute(0, 2, 1)  # [B, N_patches, D]
#         return x