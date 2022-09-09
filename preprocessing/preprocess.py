import logging
import random
import typing as tp

from transformers import BertTokenizer

from data import Dataset


class Preprocessor:
    def __init__(
        self, dataset: Dataset, max_length: int = 100, limit: tp.Optional[int] = None
    ):
        # we need to implement the search strategy
        self._device_: str = "cpu"
        self._dataset_ = dataset
        self._max_length_: int = max_length
        self._limit_ = limit if limit is not None else dataset.len
        self._tokenizer_ = BertTokenizer.from_pretrained(
            "bert-base-multilingual-uncased"
        )

        langs = ["en", "it", "es", "de", "fr"]
        langs_index = {}
        self._tokenizer_.add_tokens([f"[2{lang}]" for lang in langs])
        for lang in langs:
            langs_index[lang] = self._tokenizer_.get_added_vocab()[f"[2{lang}]"]
        self._tokenizer_.add_special_tokens({"pad_token": "[PAD]"})
        self._pad_index_ = self._tokenizer_.vocab["[PAD]"]

    def _preprocessing_(self, train_strings):
        train_data = []
        for src, trg in train_strings:
            src = (
                self._tokenizer_(
                    src,
                    return_tensors="pt",
                    add_special_tokens=True,
                    max_length=self._max_length_,
                    padding="max_length",
                    truncation=True,
                )
                .to(self._device_)
                .data["input_ids"]
            )
            trg = (
                self._tokenizer_(
                    trg,
                    return_tensors="pt",
                    add_special_tokens=True,
                    max_length=self._max_length_,
                    padding="max_length",
                    truncation=True,
                )
                .to(self._device_)
                .data["input_ids"]
            )
            train_data.append((src, trg))
        return train_data

    def _wrap_preprocessing_by_lang_(self, langs):
        def replace_lang(lang, k, v):
            return k if lang == "en" else v[lang]

        train_data = []
        for src, trg in langs:
            train_strings = [
                (f"[2{trg}] {replace_lang(src, k, v)}", replace_lang(trg, k, v))
                for k, v in self._dataset_.data.items()
            ][: self._limit_]
            train_data += self._preprocessing_(train_strings)
        random.shuffle(train_data)
        return train_data

    def execute(self):
        train_data = self._wrap_preprocessing_by_lang_(
            [
                ("en", "fr"),
                ("en", "de"),
                ("en", "es"),
                ("fr", "it"),
                ("de", "it"),
                ("es", "it"),
            ]
        )

        val_data = self._wrap_preprocessing_by_lang_([("en", "it")])

        logging.info(f"Train data: {len(train_data)}, Validation Data: {len(val_data)}")
