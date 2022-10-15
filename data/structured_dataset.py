from typing import List

from torchmetrics import SacreBLEUScore
from torchtext.data import bleu_score
import logging as lg


class TranslatedSet:
    """Class able to provide methods to evaluate the goodness of the model on NMT."""

    def __init__(self, trgs: List[str], pred: List[str], trgs_tokens: List[List[str]],
                 pred_tokens: List[List[str]]) -> None:
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

        self.__bleu__ = None
        self.__sacrebleu__ = None

    @property
    def bleu(self) -> float:
        """Method used to compute bleu score.

        Returns:
            float: BLEU score.
        """
        if self.__bleu__ is None:
            self.__bleu__ = bleu_score(self.pred_tokens, self.trgs_tokens)
        return self.__bleu__

    @property
    def sacrebleu(self) -> float:
        """Method used to compute sacrebleu score.

        Returns:
            float: sacreBLEU score.
        """
        if self.__sacrebleu__ is None:
            metric = SacreBLEUScore()
            self.__sacrebleu__ = metric(self.pred, self.trgs).item()
        return self.__sacrebleu__


class StructuredData:
    def __init__(self, labels, tokens_id):
        self.labels = labels
        self.tokens_id = tokens_id

        self.loss = None
        self.accuracy = None
        self.translated_set: TranslatedSet = None
        self.__bleu__ = None
        self.__sacrebleu__ = None

    @property
    def bleu(self) -> float:
        return self.translated_set.bleu if self.translated_set is not None else None

    @property
    def sacrebleu(self) -> float:
        return self.translated_set.sacrebleu if self.translated_set is not None else None


class TRTS:
    def __init__(self, trts, thresh_perd):
        labels, tokens_id = trts
        tr_labels, ts_labels = self.__holdout__(labels, thresh_perd)
        tr_tokens_id, ts_tokens_id = self.__holdout__(tokens_id, thresh_perd)
        self.train = StructuredData(tr_labels, tr_tokens_id)
        self.test = StructuredData(ts_labels, ts_tokens_id)

    def __holdout__(self, data, thresh_perd):
        train_data_size = len(data)
        threshold = int(train_data_size - train_data_size * thresh_perd)
        TR_SET = data[:threshold]
        TS_SET = data[threshold:]
        return TR_SET, TS_SET


class StructuredDataset:
    def __init__(self, baseset, zeroshotset, thresh_perd):
        self.__baseset__ = TRTS(baseset, thresh_perd)
        self.__zeroshotset__ = TRTS(zeroshotset, thresh_perd)

    @property
    def baseset(self):
        return self.__baseset__

    @property
    def zeroshotset(self):
        return self.__zeroshotset__

    def model_callback(self, translator, zeroshot=True, accuracy=True, bleu=True):
        def callback(trainer, epoch, train_loss, test_loss, timer):
            self.__baseset__.train.loss = train_loss
            self.__baseset__.test.loss = test_loss
            if zeroshot:
                self.compute_zeroshot_loss(trainer)
            if accuracy:
                self.compute_accuracy(trainer, zeroshot)
            if bleu:
                self.compute_bleu(translator, zeroshot)

        return callback

    def compute_zeroshot_loss(self, trainer):
        lg.info(f"Start zeroshot train loss evaluation")
        self.__zeroshotset__.train.loss = trainer.evaluate_loss(self.__zeroshotset__.train.tokens_id)
        lg.info(f"Stop zeroshot train loss evaluation")
        lg.info(f"Start zeroshot test loss evaluation")
        self.__zeroshotset__.test.loss = trainer.evaluate_loss(self.__zeroshotset__.test.tokens_id)
        lg.info(f"Stop zeroshot test loss evaluation")

    def compute_accuracy(self, trainer, zeroshot=True):
        lg.info(f"Start baseset train accuracy evaluation")
        self.__baseset__.train.accuracy = trainer.evaluate_metric(self.__baseset__.train.tokens_id)
        lg.info(f"Stop baseset train accuracy evaluation")
        lg.info(f"Start baseset test accuracy evaluation")
        self.__baseset__.test.accuracy = trainer.evaluate_metric(self.__baseset__.test.tokens_id)
        lg.info(f"Stop baseset test accuracy evaluation")
        if zeroshot:
            lg.info(f"Start zeroshot train accuracy evaluation")
            self.__zeroshotset__.train.accuracy = trainer.evaluate_metric(self.__zeroshotset__.train.tokens_id)
            lg.info(f"Stop zeroshot train accuracy evaluation")
            lg.info(f"Start zeroshot test accuracy evaluation")
            self.__zeroshotset__.test.accuracy = trainer.evaluate_metric(self.__zeroshotset__.test.tokens_id)
            lg.info(f"Stop zeroshot test accuracy evaluation")

    def compute_bleu(self, translator, zeroshot=True):
        lg.info(f"Start baseset train create_translatedset evaluation")
        self.__baseset__.train.translated_set = translator.create_translatedset(self.__baseset__.train.labels)
        lg.info(f"Stop baseset train create_translatedset evaluation")
        lg.info(f"Start baseset test create_translatedset evaluation")
        self.__baseset__.test.translated_set = translator.create_translatedset(self.__baseset__.test.labels)
        lg.info(f"Stop baseset test create_translatedset evaluation")
        if zeroshot:
            lg.info(f"Start zeroshot train create_translatedset evaluation")
            self.__zeroshotset__.train.translated_set = translator.create_translatedset(self.__zeroshotset__.train.labels)
            lg.info(f"Stop zeroshot train create_translatedset evaluation")
            lg.info(f"Start zeroshot test create_translatedset evaluation")
            self.__zeroshotset__.test.translated_set = translator.create_translatedset(self.__zeroshotset__.test.labels)
            lg.info(f"Stop zeroshot test create_translatedset evaluation")

    def sizes(self):
        return f"baseset\n\ttrain: {len(self.baseset.train.tokens_id)}\n\ttest: {len(self.baseset.test.tokens_id)}" \
               f"\nzeroshot\n\ttrain: {len(self.zeroshotset.train.tokens_id)}\n\ttest: {len(self.zeroshotset.test.tokens_id)}"

    def to_dict(self, trainer, translator):
        lg.info(f"Start baseset train loss evaluation")
        self.__baseset__.train.loss = trainer.evaluate_loss(self.__baseset__.train.tokens_id)
        lg.info(f"Stop baseset train loss evaluation")
        lg.info(f"Start baseset test loss evaluation")
        self.__baseset__.test.loss = trainer.evaluate_loss(self.__baseset__.test.tokens_id)
        lg.info(f"Stop baseset test loss evaluation")
        self.compute_zeroshot_loss(trainer)
        self.compute_accuracy(trainer)
        self.compute_bleu(translator)
        return {
            'baseset': {
                'train': {
                    'loss': self.__baseset__.train.loss,
                    'accuracy': self.__baseset__.train.accuracy,
                    'bleu': self.__baseset__.train.bleu,
                    'sacrebleu': self.__baseset__.train.sacrebleu,
                },
                'test': {
                    'loss': self.__baseset__.test.loss,
                    'accuracy': self.__baseset__.test.accuracy,
                    'bleu': self.__baseset__.test.bleu,
                    'sacrebleu': self.__baseset__.test.sacrebleu,
                },
            },
            'zeroshot': {
                'train': {
                    'loss': self.__zeroshotset__.train.loss,
                    'accuracy': self.__zeroshotset__.train.accuracy,
                    'bleu': self.__zeroshotset__.train.bleu,
                    'sacrebleu': self.__zeroshotset__.train.sacrebleu,
                },
                'test': {
                    'loss': self.__zeroshotset__.test.loss,
                    'accuracy': self.__zeroshotset__.test.accuracy,
                    'bleu': self.__zeroshotset__.test.bleu,
                    'sacrebleu': self.__zeroshotset__.test.sacrebleu,
                },
            }
        }
