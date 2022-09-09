import logging
import sys

from data import Dataset, DatasetDownloader
from preprocessing import Preprocessor

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
dataset_downloader = DatasetDownloader()
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
preprocessor = Preprocessor(dataset=dataset, max_length=100, limit=None)

preprocessor.execute()
