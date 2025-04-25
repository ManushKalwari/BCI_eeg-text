import torch
import torch.nn as nn



class LearnablePositionalEncoding(nn.Module):

    # add a learnable vector to each patch based on its position.

    def __init__(self, max_len, embed_dim):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x: [batch, seq_len, embed_dim]
        return x + self.pos_embed[:, :x.size(1), :]
