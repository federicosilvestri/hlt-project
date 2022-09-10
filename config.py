"""Configuration file"""
from pathlib import Path

# Dataset settings
DATASET_URL = "https://github.com/federicosilvestri/hlt-parallel-dataset/blob/master/processed/parallel.json?raw=true"
DATASET_DOWNLOAD_DIR = Path(__file__).parent / "dataset"
DATASET_FILE_NAME = "dataset.json"

# Preprocessor Serializer
PREPROCESSOR_DIR = Path(__file__).parent / "serialized"
PREPROCESSOR_FILE_NAME = "preprocessor.pickle"
