from model.layers.positional_embeddings import PositionalEmbedding
import torch
from torch import nn
from model.layers.encoder_layer import EncoderLayer


class Encoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 max_length=100):
        super().__init__()

        self.device = device

        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = PositionalEmbedding(max_length, hid_dim, device)

        self.layers = nn.ModuleList([EncoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask = None):
        # src = [batch size, src len]
        src = src.to(self.device)
        src_len = src.shape[1]
        # pos = [batch size, src len]
        src = self.dropout((self.tok_embedding(src)) + self.pos_embedding(src_len))
        # src = [batch size, src len, hid dim]
        for layer in self.layers:
            src = layer(src, src_mask)
        # src = [batch size, src len, hid dim]
        return src
