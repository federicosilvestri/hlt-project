"""Configuration file"""
from pathlib import Path
import random
import torch
import numpy as np
from utils import search_strategy

GENERATED_FILE_DIR = Path(__file__).parent / "generated"
if not GENERATED_FILE_DIR.exists():
    GENERATED_FILE_DIR.mkdir()

# Dataset settings
DATASET_URL = "https://github.com/federicosilvestri/hlt-parallel-dataset/blob/master/processed/parallel.json?raw=true"
DATASET_DOWNLOAD_DIR = GENERATED_FILE_DIR / "dataset"
DATASET_FILE_NAME = "dataset.json"

# Preprocessor Serializer
PREPROCESSOR_DIR = GENERATED_FILE_DIR / "serialized"
PREPROCESSOR_FILE_NAME = "preprocessor.pickle"
CHUNKS = 50

# Random setting
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Device
DEVICE = search_strategy()

# Model configuration
PRETRAINED = False
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
BATCH_SIZE = 32
EPOCHS = 1

# Translator configuration
MAX_LENGTH = 100

# Plot configuration
PLOTS_DIR = GENERATED_FILE_DIR / "plots"
LOSS_PLOT_FILE_NAME = "loss_plot.png"