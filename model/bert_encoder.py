from transformers import BertConfig, BertModel

from pathlib import Path
import torch
from torch import nn


class BERTEncoder(nn.Module):
    def __init__(self,
                 hid_dim,
                 pf_dim,
                 tokenizer_dim,
                 device,
                 ):
        super().__init__()

        bert_config = BertConfig.from_pretrained('bert-base-multilingual-uncased')

        bert_config.hidden_size = hid_dim
        bert_config.num_attention_heads = pf_dim

        self.bert = BertModel(bert_config).to(device)
        self.bert.resize_token_embeddings(tokenizer_dim)

    def forward(self, src):
        src = src.view(src.shape[0], -1)
        out = self.bert(src)
        return out[0]
