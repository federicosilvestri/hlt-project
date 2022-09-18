from typing import List, Tuple
import torch
from torch import nn
from torchtext.data.metrics import bleu_score
from torchmetrics import SacreBLEUScore
from transformers import AutoTokenizer

from utils import search_strategy


class TranslatedSet:
    """Class able to provide methods to evaluate the goodness of the model on NMT."""

    def __init__(self, trgs: List[str], pred: List[str], trgs_tokens: List[List[str]], pred_tokens: List[List[str]]) -> None:
        """TranslatedSet constructor.

        Args:
            trgs (List[str]): Targets.
            pred (List[str]): Predictions.
            trgs_tokens (List[List[str]]): Tokens of targets.
            pred_tokens (List[List[str]]): Tokens of predictions.
        """
        self.trgs = trgs
        self.pred = pred
        self.trgs_tokens = trgs_tokens
        self.pred_tokens = pred_tokens

    def bleu(self) -> float:
        """Method used to compute bleu score.

        Returns:
            float: BLEU score.
        """
        return bleu_score(self.pred_tokens, self.trgs_tokens)

    def sacre_bleu(self) -> float:
        """Method used to compute sacre bleu score.

        Returns:
            float: sacre BLEU score.
        """
        metric = SacreBLEUScore()
        return metric(self.pred, self.trgs)


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
        trg_indexes = self.tokenizer_decoder.convert_tokens_to_ids(["[CLS]"])
        for _ in range(self.max_length):
            trg_tensor = torch.tensor(trg_indexes).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output, _ = self.model(src_tensor, trg_tensor)
            pred_token = output.argmax(2)[:, -1].item()
            trg_indexes.append(pred_token)
            trg_tokens = self.tokenizer_decoder.convert_ids_to_tokens(trg_indexes)
            if trg_tokens[-1] == "[SEP]":
                break
        trg_tokens = trg_tokens[1:-1]
        res_sent = self.tokenizer_decoder.convert_tokens_to_string(trg_tokens)
        return res_sent

    def create_translatedset(self, test_set: List[Tuple[str, str]]) -> TranslatedSet:
        """Method able to create a TranslatedSet.

        Args:
            test_set (List[Tuple[str, str]]): Test set of sentances to create the translated set.

        Return:
            TranslatedSet: Translated set.
        """
        trgs = []
        trgs_tokens = []
        pred = []
        pred_tokens = []
        for src, trg in test_set:
            trgs.append([trg])
            trgs_tokens.append([self.tokenizer_decoder.tokenize(trg)])
            prediction = self.__call__(src)
            pred.append(prediction)
            pred_tokens.append(self.tokenizer_decoder.tokenize(prediction))
        return TranslatedSet(trgs, pred, trgs_tokens, pred_tokens)
