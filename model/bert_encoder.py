from transformers import BertConfig, BertModel

from torch import nn


class BERTEncoder(nn.Module):
    def __init__(self,
                 tokenizer_dim,
                 hid_dim=768,
                 n_layers=12,
                 n_heads=12,
                 pf_dim=3072,
                 dropout=0.1,
                 device='cpu',
                 type='bert-base-multilingual-uncased'
                 ):
        super().__init__()
        config = BertConfig.from_pretrained(type)
        config.hidden_size = hid_dim
        config.num_hidden_layers = n_layers
        config.num_attention_heads = n_heads
        config.intermediate_size = pf_dim
        config.hidden_dropout_prob = dropout
        self.bert = BertModel(config).to(device)
        self.bert.resize_token_embeddings(tokenizer_dim)
        self.device = device

    def forward(self, src, src_mask=None):
        src = src.to(self.device)
        out = self.bert(src, src_mask)
        return out[0]