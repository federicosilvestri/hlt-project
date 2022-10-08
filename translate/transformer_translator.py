from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple
import torch
from torch import nn
from transformers import AutoTokenizer
import numpy as np

from data.structured_dataset import TranslatedSet


class TransformerTranslator:
    """Class able to construct object that translate sentences given a transformer model."""

    def __init__(
            self,
            model: nn.Module,
            tokenizer_encoder: AutoTokenizer,
            tokenizer_decoder: AutoTokenizer,
            max_length: int = 100,
            chunks: int = None,
            device: str = 'cpu',
            limit_bleu = None
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
        self.chunks = chunks
        self.device = device
        self.limit_bleu = limit_bleu

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
        src_tensor = src_indexes.to(self.device)
        trg_indexes = self.tokenizer_decoder.convert_tokens_to_ids(["[CLS]"])
        with torch.no_grad():
            enc_src = self.model.encoder(src_tensor)
        for _ in range(self.max_length):
            trg_tensor = torch.tensor(trg_indexes).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output, attention = self.model.decoder(trg_tensor, enc_src)
            pred_index = output.argmax(-1)[:, -1].item()
            trg_indexes.append(pred_index)
            trg_tokens = self.tokenizer_decoder.convert_ids_to_tokens(trg_indexes)
            if trg_tokens[-1] == "[SEP]":
                break
        #trg_tokens = trg_tokens[1:-1]
        res_sent = self.tokenizer_decoder.convert_tokens_to_string(trg_tokens)
        return res_sent

    def compute_test_set_chunk(self, test_set):
        trgs, preds, trgs_tokens, preds_tokens = [], [], [], []
        for src, trg in test_set:
            trg_tokens = self.tokenizer_decoder.tokenize(trg)
            pred = self.__call__(src)
            pred_tokens = self.tokenizer_decoder.tokenize(pred)
            trgs.append([trg])
            preds.append(pred)
            trgs_tokens.append([trg_tokens])
            preds_tokens.append(pred_tokens)
        return trgs, preds, trgs_tokens, preds_tokens

    def create_translatedset(self, test_set: List[Tuple[str, str]]) -> TranslatedSet:
        """Method able to create a TranslatedSet.

        Args:
            test_set (List[Tuple[str, str]]): Test set of sentances to create the translated set.

        Return:
            TranslatedSet: Translated set.
        """
        trgs, preds, trgs_tokens, preds_tokens = [], [], [], []
        if self.chunks is not None:
            test_set = np.array_split(test_set[:self.limit_bleu], self.chunks)

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.compute_test_set_chunk, chunk) for chunk in test_set]
            for future in futures:
                trg_chunk, pred_chunk, trg_tokens_chunk, pred_tokens_chunk = future.result()
                trgs += trg_chunk
                preds += pred_chunk
                trgs_tokens += trg_tokens_chunk
                preds_tokens += pred_tokens_chunk

        return TranslatedSet(trgs, preds, trgs_tokens, preds_tokens)
