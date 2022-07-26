import pickle
from pathlib import Path


class SDSerializer:
    def __init__(self, file_dir: Path, file_name: str):
        self.__file_dir__: Path = file_dir
        self.__file_name__: str = file_name
        self.__file_path__: Path = file_dir / file_name

    def exists(self) -> bool:
        """
        Check if structured dataset has been saved before or not.
        Returns: True if exists the file, false if not
        """
        return self.__file_path__.exists()

    def serialize(self, obj):
        if not self.__file_dir__.exists():
            self.__file_dir__.mkdir()

        with open(self.__file_path__, "wb") as fp:
            pickle.dump(obj=obj, file=fp)

    def load(self):
        with open(self.__file_path__, "rb") as fp:
            obj = pickle.load(fp)

        return obj
