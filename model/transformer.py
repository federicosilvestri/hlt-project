from pathlib import Path
import torch
from torch import nn


class Transformer(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 src_pad_idx,
                 trg_pad_idx,
                 device,
                 model_dir: Path,
                 model_file_name: str
                 ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        self.model_dir = model_dir
        self.model_file_name = model_file_name

    def make_trg_mask(self, trg):
        # trg = [batch size, trg len]
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        # trg_pad_mask = [batch size, 1, 1, trg len]
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(
            (trg_len, trg_len), device=self.device)).bool()
        # trg_sub_mask = [trg len, trg len]
        trg_mask = trg_pad_mask & trg_sub_mask
        # trg_mask = [batch size, 1, trg len, trg len]
        return trg_mask

    def forward(self, src, trg):
        # src = [batch size, src len]
        # trg = [batch size, trg len]
        trg_mask = self.make_trg_mask(trg)
        # src_mask = [batch size, 1, 1, src len]
        # trg_mask = [batch size, 1, trg len, trg len]
        enc_src = self.encoder(src)
        # enc_src = [batch size, src len, hid dim]
        output, attention = self.decoder(trg, enc_src, trg_mask)
        # output = [batch size, trg len, output dim]
        # attention = [batch size, n heads, trg len, src len]
        return output, attention

    def save_transformer(self):
        if not self.model_dir.exists():
            self.model_dir.mkdir()
        torch.save(self, self.model_dir / self.model_file_name)
