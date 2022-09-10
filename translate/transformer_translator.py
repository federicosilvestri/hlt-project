from typing import List, Tuple
import torch
from torch import nn
from torchtext.data.metrics import bleu_score
from transformers import AutoTokenizer

from utils import search_strategy


class TransformerTranslator:
    """Class able to construct object that translate sentences given a transformer model."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer_encoder: AutoTokenizer,
        tokenizer_decoder: AutoTokenizer,
        max_length: int = 100,
        device: str = 'cpu'
    ) -> None:
        """TransformerTranslator constructor.

        Args:
            model (nn.Module): Model trained that is able to produce a word given a source cotext and a target sentence.
            tokenizer_encoder (AutoTokenizer): Encoder tokenizer.
            tokenizer_decoder (AutoTokenizer): Decoder tokenizer.
            max_length (int, optional): Max number of tokens of a translated sentence.
            device (str, optional): Accelerator used to translate data.
        """
        self.model = model
        self.tokenizer_encoder = tokenizer_encoder
        self.tokenizer_decoder = tokenizer_decoder
        self.max_length = max_length
        self.device = device

    def __call__(self, src_sentence: str) -> str:
        """Method able to translate a given sentence.

        Args:
            src_sentence (src): Source sentence.

        Returns:
            str: Translated sentence.
        """
        src_indexes = (
            self.tokenizer_encoder(
                src_sentence,
                return_tensors="pt",
                add_special_tokens=True,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
            )
            .to(self.device)
            .data["input_ids"]
        )
        src_tensor = src_indexes.unsqueeze(0).to(self.device)
        src_mask = self.model.make_src_mask(src_tensor)
        with torch.no_grad():
            src_mask = src_mask.squeeze(1).squeeze(2)
            enc_src = self.model.encoder(src_tensor, src_mask)
        trg_indexes = self.tokenizer_decoder.convert_tokens_to_ids(["[CLS]"])
        for i in range(self.max_length):
            trg_tensor = torch.tensor(trg_indexes).unsqueeze(0).to(self.device)
            trg_mask = self.model.make_trg_mask(trg_tensor)
            with torch.no_grad():
                output, attention = self.model.decoder(
                    trg_tensor, enc_src, trg_mask, src_mask
                )
            pred_token = output.argmax(2)[:, -1].item()
            trg_indexes.append(pred_token)
            trg_tokens = self.tokenizer_decoder.convert_ids_to_tokens(trg_indexes)
            if trg_tokens[-1] == "[SEP]":
                break
        trg_tokens = trg_tokens[1:-1]
        res_sent = self.tokenizer_decoder.convert_tokens_to_string(trg_tokens)
        return res_sent

    def bleu(self, test_set: List[Tuple[str, str]]) -> float:
        """Method used to compute bleu score.

        Args:
            test_set (List[Tuple[torch.tensor, torch.tensor]]): Test set.

        Returns:
            float: BLEU score.
        """
        trgs = []
        pred_trgs = []
        for src, trg in test_set:
            trg_tokens = self.tokenizer_decoder.tokenize(trg)
            trgs.append([trg_tokens])
            pred_trg = self.__call__(src)
            pred_trg_tokens = self.tokenizer_decoder.tokenize(pred_trg)
            pred_trgs.append(pred_trg_tokens)
        return bleu_score(pred_trgs, trgs)
