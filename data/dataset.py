import json
import logging
import typing as tp
from pathlib import Path
import random


class Dataset:
    """This class represents the Dataset"""

    def __init__(self, file: Path, cut=1):
        self.__ds_file__ = file
        self.__ds__: tp.List[tp.Dict[str, str]]

        logging.info("Loading dataset inside memory")
        # loading the file inside the memory
        with open(self.__ds_file__) as fp:
            dataset = json.load(fp)
            keys = [key for key in dataset.keys()]
            random.shuffle(keys)
            keys = keys[:len(keys) // cut]
            self.__ds__ = {key: dataset[key] for key in keys}

    @property
    def data(self):
        return self.__ds__

    @property
    def size(self):
        return len(self.__ds__)
