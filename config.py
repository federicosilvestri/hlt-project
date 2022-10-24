"""Configuration file"""
import os
from pathlib import Path
import random
import torch
import numpy as np
from utils import search_strategy
from transformers import BertTokenizer

GENERATED_FILE_DIR = Path(__file__).parent / "generated"
if not GENERATED_FILE_DIR.exists():
    GENERATED_FILE_DIR.mkdir()

# Dataset settings
DATASET_URL = "https://github.com/federicosilvestri/hlt-parallel-dataset/blob/master/processed/parallel.json?raw=true"
DATASET_DOWNLOAD_DIR = GENERATED_FILE_DIR / "dataset"
DATASET_FILE_NAME = "dataset.json"
DATASET_CUT = 12

# Preprocessor Serializer
TOKENIZER = BertTokenizer.from_pretrained("bert-base-multilingual-uncased")
TOKENIZER.add_tokens([f"[2{lang}]" for lang in ["en", "it", "es", "de", "fr"]])
VOCAB_SIZE = len(TOKENIZER)
PREPROCESSOR_DIR = GENERATED_FILE_DIR / "serialized"
PREPROCESSOR_FILE_NAME = "preprocessor.pickle"
BASE_LANG_CONFIG = [
    ("en", "it"),
    ("en", "es"),
    ("en", "de"),
    ("en", "fr"),
    ("it", "en"),
    ("es", "en"),
    ("de", "en"),
    ("fr", "en"),
]
ZEROSHOT_LANG_CONFIG = [
    ("fr", "it"),
    ("fr", "es"),
    ("fr", "de"),
    ("it", "fr"),
    ("es", "fr"),
    ("de", "fr"),
    ("it", "es"),
    ("it", "de"),
    ("es", "it"),
    ("de", "it"),
    ("es", "de"),
    ("de", "es"),
]

# Random setting
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Device
DEVICE = search_strategy()
N_DEGREE = os.cpu_count()

# Model configuration
PRETRAINED_TYPE = None
HID_DIM = 256
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1
MODEL_DIR = GENERATED_FILE_DIR / "model"
MODEL_FILE_NAME = "zero_shot_nmt.torch"

# Trainer configuration
HOLDOUT_VALID_FRACTION = 0.1
BATCH_SIZE = 128
EPOCHS = 10
CLIP = 1
LEARNING_RATE = 0.0005
LIMIT_EVAL = None

# Translator configuration
MAX_LENGTH = 100
LIMIT_BLEU = 100

# Plot configuration
PLOTS_DIR = GENERATED_FILE_DIR / "plots"
