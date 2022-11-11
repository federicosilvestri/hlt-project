from transformers import DistilBertConfig, DistilBertModel

from torch import nn


class DistilBERTEncoder(nn.Module):
    def __init__(self,
                 tokenizer_dim,
                 hid_dim=768,
                 n_layers=6,
                 n_heads=12,
                 pf_dim=3072,
                 dropout=0.1,
                 device='cpu',
                 type='distilbert-base-multilingual-uncased'
                 ):
        super().__init__()
        config = DistilBertConfig.from_pretrained(type)
        config.dim = hid_dim
        config.n_layers = n_layers
        config.n_heads = n_heads
        config.hidden_dim = pf_dim
        config.dropout = dropout
        self.bert = DistilBertModel(config).to(device)
        self.bert.resize_token_embeddings(tokenizer_dim)
        self.device = device

    def forward(self, src, src_mask=None):
        src = src.to(self.device)
        out = self.bert(src, src_mask)
        return out[0]
