import typing as tp
from data import Dataset


class Preprocessor:
    def __init__(
            self, dataset: Dataset, limit: tp.Optional[int] = None,
    ):
        # we need to implement the search strategy
        self._dataset_ = dataset
        self._limit_ = limit if limit is not None else dataset.size

    def _wrap_preprocessing_by_lang_(self, langs):
        def replace_lang(lang, k, v):
            return k if lang == "en" else v[lang]

        train_strings = []
        for k, v in list(self._dataset_.data.items())[: self._limit_]:
            train_strings += [
                                 (f"[2{trg}] {replace_lang(src, k, v)}", f"{replace_lang(trg, k, v)}")
                                 for src, trg in langs
                             ]

        return train_strings

    def execute(self, base_lang_config, zeroshot_lang_config):
        base_data = self._wrap_preprocessing_by_lang_(base_lang_config)
        zeroshot_data = self._wrap_preprocessing_by_lang_(zeroshot_lang_config)
        return base_data, zeroshot_data
