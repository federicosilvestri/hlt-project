import logging
import sys

from config import (
    DATASET_DOWNLOAD_DIR,
    DATASET_FILE_NAME,
    DATASET_URL,
    PREPROCESSOR_DIR,
    PREPROCESSOR_FILE_NAME,
)
from data import Dataset, DatasetDownloader
from preprocessing import Preprocessor, PreprocessSerializer

root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
root.addHandler(handler)

#
# Downloading dataset
#
logging.info("Executing pipeline")
dataset_downloader = DatasetDownloader(
    download_dir=DATASET_DOWNLOAD_DIR, url=DATASET_URL, file_name=DATASET_FILE_NAME
)

if not dataset_downloader.already_downloaded():
    logging.info("Downloading dataset")
    dataset_downloader.download()

#
# Creating the dataset
#
logging.info("Loading dataset")
dataset = Dataset(dataset_downloader.downloaded_file)

#
# Execute preprocessing
#
preprocessor_serializer = PreprocessSerializer(
    file_name=PREPROCESSOR_FILE_NAME, file_dir=PREPROCESSOR_DIR
)

if not preprocessor_serializer.exists():
    logging.info("Preprocessing file not found, executing preprocessing...")
    preprocessor = Preprocessor(dataset=dataset, max_length=100, limit=None)
    # executing preprocessing
    preprocessor.execute()

    # saving
    logging.info("Saving preprocessor into file")
    preprocessor_serializer.serialize(preprocessor)
else:
    logging.info("Loading preprocessor from file")
    preprocessor = preprocessor_serializer.load()
