from model.layers.positional_embeddings import PositionalEmbedding
import torch
from torch import nn
from model.layers.decoder_layer import DecoderLayer


class Decoder(nn.Module):
    def __init__(self,
                 output_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 max_length=100):
        super().__init__()

        self.device = device

        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = PositionalEmbedding(max_length, hid_dim, device)

        self.layers = nn.ModuleList([DecoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, src_mask = None, trg_mask = None):
        # trg = [batch size, trg len]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, 1, trg len, trg len]
        trg = trg.to(self.device)
        trg_len = trg.shape[1]
        # pos = [batch size, trg len]
        trg = self.dropout(
            (self.tok_embedding(trg)) + self.pos_embedding(trg_len))
        # trg = [batch size, trg len, hid dim]
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, src_mask, trg_mask)
        # trg = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]
        output = self.fc_out(trg)
        # output = [batch size, trg len, output dim]
        return output, attention
