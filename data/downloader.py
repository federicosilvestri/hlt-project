import logging
from pathlib import Path

import wget

# Parallel dataset built with another repository
DATASET_URL = "https://github.com/federicosilvestri/hlt-parallel-dataset/blob/master/processed/parallel.json?raw=true"
DOWNLOAD_DIR = Path(__file__).parent.parent / "dataset"
FILE_NAME = "dataset.json"


class DatasetDownloader:
    """
    Dataset downloader class that uses wget function to decrease memory utilization.
    """

    def __init__(
        self,
        url: str = DATASET_URL,
        download_dir: Path = DOWNLOAD_DIR,
        file_name: str = FILE_NAME,
    ):
        self.__url__: str = url
        self.__download_dir__: Path = download_dir
        self.__file_name__: str = file_name

    def already_downloaded(self) -> bool:
        """
        Check if dataset is already downloaded
        Returns: True if it's already downloaded, else False.

        """
        if not self.__download_dir__.exists():
            return False

        for element in self.__download_dir__.iterdir():
            if element.name == self.__file_name__:
                return True

        return False

    @property
    def downloaded_file(self) -> Path:
        return self.__download_dir__ / self.__file_name__

    def download(self) -> None:
        """
        Download the dataset.
        Returns: None

        """
        download_path = self.__download_dir__.resolve()
        if not download_path.exists():
            logging.info("Dataset does not exist, creating directory")
            download_path.mkdir()

        logging.info("Starting the download of dataset")

        # download the dataset
        out_file = wget.download(
            url=self.__url__,
            out=download_path.absolute().__str__(),
            bar=wget.bar_adaptive,
        )

        out_file_path: Path = Path(out_file).resolve()
        new_out_file = download_path.joinpath(self.__file_name__)
        out_file_path.rename(new_out_file)

        logging.info("Dataset downloaded")
        return None
