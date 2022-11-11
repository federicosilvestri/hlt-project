from transformers import MT5Config, MT5EncoderModel

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
        config = MT5Config.from_pretrained(type)
        config.d_model = hid_dim
        config.num_layers = n_layers
        config.num_heads = n_heads
        config.d_ff = pf_dim
        config.dropout_rate = dropout
        self.mt5 = MT5EncoderModel(config).to(device)
        self.mt5.resize_token_embeddings(tokenizer_dim)
        self.device = device

    def forward(self, src, src_mask=None):
        src = src.to(self.device)
        out = self.mt5(src, src_mask)
        return out[0]
