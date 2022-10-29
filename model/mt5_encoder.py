from transformers import MT5Config, MT5EncoderModel

from pathlib import Path
import torch
from torch import nn


class MT5Encoder(nn.Module):
    def __init__(self,
                 tokenizer_dim,
                 hid_dim=512,
                 n_layers=8,
                 n_heads=6,
                 pf_dim=1024,
                 dropout=0.1,
                 device='cpu',
                 type='google/mt5-small'
                 ):
        super().__init__()
        mt5_config = MT5Config.from_pretrained(type)
        mt5_config.d_model = hid_dim
        mt5_config.num_layers = n_layers
        mt5_config.num_heads = n_heads
        mt5_config.d_ff = pf_dim
        mt5_config.dropout_rate = dropout
        self.mt5 = MT5EncoderModel(mt5_config).to(device)
        self.mt5.resize_token_embeddings(tokenizer_dim)
        self.device = device

    def forward(self, src, src_mask=None):
        src = src.to(self.device)
        out = self.mt5(src, src_mask)
        return out[0]
