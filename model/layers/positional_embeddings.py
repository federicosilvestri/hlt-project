import math
import torch
from torch import nn


class PositionalEmbedding(nn.Module):
    def __init__(self, max_length, embed_dim, device):
        super(PositionalEmbedding, self).__init__()
        self.max_length = max_length
        self.embed_dim = embed_dim
        positional_embedding = torch.zeros(max_length, embed_dim, requires_grad=False).to(device)
        for pos in range(max_length):
            for i in range(0, embed_dim, 2):
                arg = pos / math.pow(10000, (i * 2) / embed_dim)
                positional_embedding[pos, i] = math.sin(arg)
                positional_embedding[pos, i + 1] = math.cos(arg)
        positional_embedding = positional_embedding.unsqueeze(0)
        self.positional_embedding = positional_embedding

    def forward(self, seq_len):
        return self.positional_embedding[:, :seq_len]
