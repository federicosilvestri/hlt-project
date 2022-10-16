from transformers import BertConfig, BertModel

from pathlib import Path
import torch
from torch import nn


class BERTEncoder(nn.Module):
    def __init__(self,
                 hid_dim,
                 tokenizer_dim,
                 device,
                 type='bert-base-multilingual-cased'
                 ):
        super().__init__()
        bert_config = BertConfig.from_pretrained(type)
        bert_config.hidden_size = hid_dim
        self.bert = BertModel(bert_config).to(device)
        self.bert.resize_token_embeddings(tokenizer_dim)
        self.device = device

    def forward(self, src, src_mask=None):
        src = src.to(self.device)
        out = self.bert(src, src_mask)
        return out[0]
