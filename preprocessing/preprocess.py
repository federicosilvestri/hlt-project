from concurrent.futures import ThreadPoolExecutor
from functools import reduce
import typing as tp
import numpy as np
from data import Dataset
from utils import search_strategy


class Preprocessor:
    def __init__(
            self, dataset: Dataset, tokenizer, device, max_length: int = 100, limit: tp.Optional[int] = None,
            chunks: int = None
    ):
        # we need to implement the search strategy
        self._device_: str = device
        self._dataset_ = dataset
        self._max_length_: int = max_length
        self._limit_ = limit if limit is not None else dataset.size
        self.chunks = chunks
        self._tokenizer_ = tokenizer

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

        train_strings = []
        for k, v in self._dataset_.data.items():
            train_strings += [
                                 (f"[2{trg}] {replace_lang(src, k, v)}", f"{replace_lang(trg, k, v)}")
                                 for src, trg in langs
                             ][: self._limit_]
        if self.chunks is not None and self.chunks > 1:
            train_strings_chunks = np.array_split(train_strings, self.chunks)
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(self._preprocessing_, train_str) for train_str in train_strings_chunks]
                train_data = reduce(lambda x, y: x + y, [future.result() for future in futures])
        else:
            train_data = self._preprocessing_(train_strings)

        return train_strings, train_data

    def execute(self, base_lang_config, zeroshot_lang_cnfig):
        base_data = self._wrap_preprocessing_by_lang_(base_lang_config)
        zeroshot_data = self._wrap_preprocessing_by_lang_(zeroshot_lang_cnfig)
        return base_data, zeroshot_data
