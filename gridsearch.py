import math
from itertools import product
import json
from threading import Thread

from config import *
from data.structured_dataset import StructuredDataset
from model.bert_encoder import BERTEncoder
from model.decoder import Decoder
from model.encoder import Encoder
from model.mt5_encoder import MT5Encoder
from model.transformer import Transformer
from trainer.trainer import Trainer
from pipeline import Pipeline
import logging as lg
import numpy as np
import os

from trainer.trainer_callbacks import print_epoch_loss_accuracy

class ModelType:
    PERSONAL=0
    BERT=1
    MT5=2


class Hyperparameters:
    HID_DIM = [768]
    ENC_LAYERS = [
        (ModelType.MT5, 'google/mt5-small'),
        (ModelType.BERT, 'bert-base-multilingual-cased'),
        (ModelType.BERT, 'distilbert-base-multilingual-cased'),
        (ModelType.PERSONAL, 3),
    ]
    DEC_LAYERS = [3]
    ENC_PF_DIM = [512]
    DEC_PF_DIM = [512]
    LEARNING_RATE = [0.0005]
    CLIP = [1]


class GridSearch:
    def __init__(self, hyperparameters: Hyperparameters):
        self.hyperparameters = hyperparameters
        self.n_chunks = min(torch.cuda.device_count(), N_DEGREE) if DEVICE == 'cuda' else N_DEGREE
        gs_dir = GENERATED_FILE_DIR / "gridsearch"
        if not gs_dir.exists():
            gs_dir.mkdir()
        self.gs_files = []
        self.gs_dicts = []
        for n_chunk in range(self.n_chunks):
            gs_file = gs_dir / f"gridsearch{n_chunk}.json"
            if gs_file.exists():
                with open(gs_file, "w") as fp:
                    fp.truncate(0)
            else:
                gs_file.touch()
            self.gs_files.append(gs_file)
            gs_dict = []
            with open(gs_file, "w") as fp:
                json.dump(gs_dict, fp)
            self.gs_dicts.append(gs_dict)

    def __train_chunk(self, n_chunk, structured_dataset: StructuredDataset, epochs, pipeline, hyperparams):
        i = 0
        print_callback = print_epoch_loss_accuracy(structured_dataset)
        device = f'{DEVICE}:{n_chunk}' if DEVICE == 'cuda' else DEVICE
        for HID_DIM, ENC_TUPLE, DEC_LAYERS, ENC_PF_DIM, DEC_PF_DIM, LEARNING_RATE, CLIP in hyperparams:
            i += 1
            lg.info(f"CHUNK {n_chunk} - Start configuration {i}/{len(hyperparams)}")

            INPUT_DIM = VOCAB_SIZE
            OUTPUT_DIM = VOCAB_SIZE

            ENC_TYPE, ENC_CONFIG = ENC_TUPLE
            if ENC_TYPE == ModelType.BERT:
                enc = BERTEncoder(HID_DIM, VOCAB_SIZE, device, type=ENC_CONFIG)
            elif ENC_TYPE == ModelType.MT5:
                enc = MT5Encoder(HID_DIM, VOCAB_SIZE, device, type=ENC_CONFIG)
            elif ENC_TYPE == ModelType.PERSONAL:
                enc = Encoder(INPUT_DIM,
                              HID_DIM,
                              ENC_CONFIG,
                              ENC_HEADS,
                              ENC_PF_DIM,
                              ENC_DROPOUT,
                              device)

            dec = Decoder(OUTPUT_DIM,
                          HID_DIM,
                          DEC_LAYERS,
                          DEC_HEADS,
                          DEC_PF_DIM,
                          DEC_DROPOUT,
                          device)

            model = Transformer(enc, dec, device, MODEL_DIR, MODEL_FILE_NAME).to(device)

            trainer = Trainer(model, TOKENIZER.vocab['pad'], LEARNING_RATE, clip=CLIP, device=device,
                              limit_eval=LIMIT_EVAL)
            translator = pipeline.create_translator(model, device=device)
            trainer(
                structured_dataset.baseset.train.tokens_id,
                structured_dataset.baseset.test.tokens_id,
                epochs=epochs, callbacks=[
                    structured_dataset.model_callback(translator),
                    print_callback
                ], save_model=False)
            gs_dict_iter = {
                "hyperparams": {
                    "HID_DIM": HID_DIM,
                    "ENC_TYPE": ENC_TYPE,
                    "ENC_CONFIG": ENC_CONFIG,
                    "DEC_LAYERS": DEC_LAYERS,
                    "ENC_PF_DIM": ENC_PF_DIM,
                    "DEC_PF_DIM": DEC_PF_DIM,
                    "LEARNING_RATE": LEARNING_RATE,
                    "CLIP": CLIP
                },
                "structured_dataset": structured_dataset.to_dict(trainer, translator)
            }
            with open(f'{self.gs_files[n_chunk]}', 'w') as fp:
                self.gs_dicts[n_chunk].append(gs_dict_iter)
                json.dump(self.gs_dicts[n_chunk], fp)

    def train(self, epochs=10):
        pipeline = Pipeline()
        dataset = pipeline.dataset_load()
        structured_dataset = pipeline.preprocess(dataset)
        lg.info(f"Structured dataset sizes\n{structured_dataset.sizes()}")
        hyperparams = list(product(
            self.hyperparameters.HID_DIM,
            self.hyperparameters.ENC_LAYERS,
            self.hyperparameters.DEC_LAYERS,
            self.hyperparameters.ENC_PF_DIM,
            self.hyperparameters.DEC_PF_DIM,
            self.hyperparameters.LEARNING_RATE,
            self.hyperparameters.CLIP,
        ))
        lg.info(f"CHUNKS {self.n_chunks} for {len(hyperparams)} configurations")
        if self.n_chunks == 1:
            self.__train_chunk(0, structured_dataset, epochs, pipeline, hyperparams)
        else:
            hyperparams_chunks = [[] for _ in range(self.n_chunks)]
            for i, hyperparam in enumerate(hyperparams):
                hyperparams_chunks[i % self.n_chunks].append(hyperparam)
            threads = []
            for n_chunk, hyperparams_chunk in enumerate(hyperparams_chunks):
                thread = Thread(
                    target=self.__train_chunk,
                    args=(n_chunk, structured_dataset, epochs, pipeline, hyperparams_chunk)
                )
                thread.start()
                threads.append(thread)
            for thread in threads:
                thread.join()


if __name__ == "__main__":
    gs = GridSearch(Hyperparameters())
    gs.train(1)
