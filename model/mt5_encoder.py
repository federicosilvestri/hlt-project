from transformers import MT5Config, MT5EncoderModel

from pathlib import Path
import torch
from torch import nn


class MT5Encoder(nn.Module):
    def __init__(self,
                 hid_dim,
                 tokenizer_dim,
                 device,
                 type='google/mt5-small'
                 ):
        super().__init__()
        mt5_config = MT5Config.from_pretrained(type)
        mt5_config.d_model = hid_dim
        self.mt5 = MT5EncoderModel(mt5_config).to(device)
        self.mt5.resize_token_embeddings(tokenizer_dim)
        self.device = device

    def forward(self, src, src_mask=None):
        src = src.to(self.device)
        out = self.mt5(src, src_mask)
        return out[0]
