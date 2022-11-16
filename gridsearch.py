import math
from itertools import product
import json
from threading import Thread

from config import *
from model.bert_encoder import BERTEncoder
from model.decoder import Decoder
from model.distilbert_encoder import DistilBERTEncoder
from model.encoder import Encoder
from model.mt5_encoder import MT5Encoder
from model.transformer import Transformer
from model.xlmroberta_encoder import XLMRobertaEncoder
from tokenizer import Tokenizer
from trainer.trainer import Trainer
from pipeline import Pipeline
import logging as lg

from trainer.trainer_callbacks import print_epoch_loss_accuracy
from transformers import BertTokenizer, DistilBertTokenizer, MT5Tokenizer, XLMRobertaTokenizer


class ModelType:
    PERSONAL = 0
    BERT = 1
    DISTILBERT = 2
    XLMROBERTA = 3
    MT5 = 4


class Hyperparameters:
    HID_DIM = [768]
    ENC_TYPES = [
        (ModelType.MT5, 'google/mt5-small'),
        (ModelType.BERT, 'bert-base-multilingual-uncased'),
        (ModelType.DISTILBERT, 'distilbert-base-multilingual-cased'),
        (ModelType.XLMROBERTA, 'xlm-roberta-base'),
        (ModelType.PERSONAL, None),
    ]
    ENC_LAYERS = [3]
    DEC_LAYERS = [3]
    ENC_PF_DIM = [512]
    DEC_PF_DIM = [512]
    LEARNING_RATE = [0.0005]
    CLIP = [1]


class GridSearch:
    def __init__(self, hyperparameters: Hyperparameters, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
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

    def __train_chunk(self, n_chunk, dataset, epochs, pipeline, hyperparams):
        i = 0
        device = f'{DEVICE}:{n_chunk}' if DEVICE == 'cuda' else DEVICE
        for HID_DIM, ENC_TYPES, ENC_LAYERS, DEC_LAYERS, ENC_PF_DIM, DEC_PF_DIM, LEARNING_RATE, CLIP in hyperparams:
            i += 1
            lg.info(f"CHUNK {n_chunk} - Start configuration {i}/{len(hyperparams)}")

            INPUT_DIM = self.tokenizer.vocab_size
            OUTPUT_DIM = self.tokenizer.vocab_size

            ENC_TYPE, ENC_MODEL_TYPE = ENC_TYPES
            if ENC_TYPE == ModelType.BERT:
                enc = BERTEncoder(INPUT_DIM,
                                  HID_DIM,
                                  ENC_LAYERS,
                                  ENC_HEADS,
                                  ENC_PF_DIM,
                                  ENC_DROPOUT,
                                  device,
                                  type=ENC_MODEL_TYPE)
            elif ENC_TYPE == ModelType.DISTILBERT:
                enc = DistilBERTEncoder(INPUT_DIM,
                                  HID_DIM,
                                  ENC_LAYERS,
                                  ENC_HEADS,
                                  ENC_PF_DIM,
                                  ENC_DROPOUT,
                                  device,
                                  type=ENC_MODEL_TYPE)
            elif ENC_TYPE == ModelType.XLMROBERTA:
                enc = XLMRobertaEncoder(INPUT_DIM,
                                  HID_DIM,
                                  ENC_LAYERS,
                                  ENC_HEADS,
                                  ENC_PF_DIM,
                                  ENC_DROPOUT,
                                  device,
                                  type=ENC_MODEL_TYPE)
            elif ENC_TYPE == ModelType.MT5:
                enc = MT5Encoder(INPUT_DIM,
                                 HID_DIM,
                                 ENC_LAYERS,
                                 ENC_HEADS,
                                 ENC_PF_DIM,
                                 ENC_DROPOUT,
                                 device,
                                 type=ENC_MODEL_TYPE)
            elif ENC_TYPE == ModelType.PERSONAL:
                enc = Encoder(INPUT_DIM,
                              HID_DIM,
                              ENC_LAYERS,
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

            structured_dataset = pipeline.preprocess(dataset, ENC_MODEL_TYPE.replace('/', '_'))
            lg.info(f"Structured dataset sizes\n{structured_dataset.sizes()}")
            print_callback = print_epoch_loss_accuracy(structured_dataset)

            trainer = Trainer(model, LEARNING_RATE, clip=CLIP, device=device,
                              limit_eval=LIMIT_EVAL)
            translator = pipeline.create_translator(model, device=device)
            trainer(
                structured_dataset.baseset.train.tokens_id,
                epochs=epochs, callbacks=[
                    structured_dataset.model_callback(translator),
                    print_callback
                ], save_model=False)
            gs_dict_iter = {
                "hyperparams": {
                    "HID_DIM": HID_DIM,
                    "ENC_TYPE": ENC_TYPE,
                    "ENC_TYPES": ENC_TYPES,
                    "ENC_LAYERS": ENC_LAYERS,
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
        pipeline = Pipeline(self.tokenizer)
        dataset = pipeline.dataset_load()
        hyperparams = list(product(
            self.hyperparameters.HID_DIM,
            self.hyperparameters.ENC_TYPES,
            self.hyperparameters.ENC_LAYERS,
            self.hyperparameters.DEC_LAYERS,
            self.hyperparameters.ENC_PF_DIM,
            self.hyperparameters.DEC_PF_DIM,
            self.hyperparameters.LEARNING_RATE,
            self.hyperparameters.CLIP,
        ))
        lg.info(f"CHUNKS {self.n_chunks} for {len(hyperparams)} configurations")
        if self.n_chunks == 1:
            self.__train_chunk(0, dataset, epochs, pipeline, hyperparams)
        else:
            hyperparams_chunks = [[] for _ in range(self.n_chunks)]
            for i, hyperparam in enumerate(hyperparams):
                hyperparams_chunks[i % self.n_chunks].append(hyperparam)
            threads = []
            for n_chunk, hyperparams_chunk in enumerate(hyperparams_chunks):
                thread = Thread(
                    target=self.__train_chunk,
                    args=(n_chunk, dataset, epochs, pipeline, hyperparams_chunk)
                )
                thread.start()
                threads.append(thread)
            for thread in threads:
                thread.join()


if __name__ == "__main__":
    tokenizer = Tokenizer(BertTokenizer.from_pretrained('bert-base-multilingual-uncased'), device=DEVICE)
    hyperparameters = Hyperparameters()
    gs = GridSearch(hyperparameters, tokenizer)
    gs.train(10)
