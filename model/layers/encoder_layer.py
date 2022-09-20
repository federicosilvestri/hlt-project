import torch
from torch import nn
from model.layers.multihead_attention_layer import MultiHeadAttentionLayer
from model.layers.feedforward_layer import FeedforwardLayer


class EncoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()

        self.self_attention = MultiHeadAttentionLayer(
            hid_dim, n_heads, dropout, device)
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.feedforward = FeedforwardLayer(hid_dim,
                                            pf_dim,
                                            dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask = None):
        # src = [batch size, src len, hid dim]

        # self attention
        _src, _ = self.self_attention(src, src, src, src_mask)

        # dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        # src = [batch size, src len, hid dim]

        # feedforward
        _src = self.feedforward(src)  # dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))
        # src = [batch size, src len, hid dim]
        return src
