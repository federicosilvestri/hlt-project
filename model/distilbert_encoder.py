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
        bert_config = DistilBertConfig.from_pretrained(type)
        bert_config.dim = hid_dim
        bert_config.n_layers = n_layers
        bert_config.n_heads = n_heads
        bert_config.hidden_dim = pf_dim
        bert_config.dropout = dropout
        self.bert = DistilBertModel(bert_config).to(device)
        self.bert.resize_token_embeddings(tokenizer_dim)
        self.device = device

    def forward(self, src, src_mask=None):
        src = src.to(self.device)
        out = self.bert(src, src_mask)
        return out[0]
