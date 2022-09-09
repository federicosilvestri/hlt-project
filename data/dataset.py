import json
import logging
import typing as tp
from pathlib import Path


class Dataset:
    """This class represents the Dataset"""

    def __init__(self, file: Path):
        self.__ds_file__ = file
        self.__ds__: tp.List[tp.Dict[str, str]]

        logging.info("Loading dataset inside memory")
        # loading the file inside the memory
        with open(self.__ds_file__) as fp:
            self.__ds__ = json.load(fp)

    @property
    def data(self):
        return self.__ds__

    @property
    def len(self):
        return len(self.__ds__)
