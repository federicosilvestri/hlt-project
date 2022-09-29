from itertools import product
import json
from config import *
from model.bert_encoder import BERTEncoder
from model.decoder import Decoder
from model.encoder import Encoder
from model.transformer import Transformer
from trainer.trainer import Trainer
from pipeline import Pipeline
import logging as lg


class Hyperparameters:
    HID_DIM = [256, 512, 768]
    ENC_LAYERS = [3, 4, 6, 8, None]
    DEC_LAYERS = [3, 4, 6, 8]
    ENC_PF_DIM = [512]
    DEC_PF_DIM = [512]
    LEARNING_RATE = [0.005, 0.001, 0.0005]
    CLIP = [1]


class GridSearch:
    def __init__(self, hyperparameters: Hyperparameters):
        self.hyperparameters = hyperparameters
        self.gs_file = GENERATED_FILE_DIR / "gridsearch.json"
        if self.gs_file.exists():
            with open(self.gs_file, "w") as fp:
                fp.truncate(0)
        else:
            self.gs_file.touch()
        self.gs_dict = []
        with open(self.gs_file, "w") as fp:
            json.dump(self.gs_dict, fp)

    def train(self, epochs=10):
        pipeline = Pipeline()
        dataset = pipeline.dataset_load()
        preprocessor = pipeline.preprocess(dataset)
        TR_SET, TS_SET = pipeline.holdout(preprocessor.trainable_data)
        ZS_TR_SET, ZS_TS_SET = pipeline.holdout(preprocessor.zeroshot_data)
        hyperparams = list(product(
            self.hyperparameters.HID_DIM,
            self.hyperparameters.ENC_LAYERS,
            self.hyperparameters.DEC_LAYERS,
            self.hyperparameters.ENC_PF_DIM,
            self.hyperparameters.DEC_PF_DIM,
            self.hyperparameters.LEARNING_RATE,
            self.hyperparameters.CLIP,
        ))
        i = 0
        for HID_DIM, ENC_LAYERS, DEC_LAYERS, ENC_PF_DIM, DEC_PF_DIM, LEARNING_RATE, CLIP in hyperparams:
            i += 1
            lg.info(f"Start configuration {i}/{len(hyperparams)}")

            INPUT_DIM = len(preprocessor._tokenizer_)
            OUTPUT_DIM = len(preprocessor._tokenizer_)

            if ENC_LAYERS is None:
                enc = BERTEncoder(HID_DIM, ENC_HEADS, len(preprocessor._tokenizer_), DEVICE)
            else:
                enc = Encoder(INPUT_DIM,
                              HID_DIM,
                              ENC_LAYERS,
                              ENC_HEADS,
                              ENC_PF_DIM,
                              ENC_DROPOUT,
                              DEVICE)

            dec = Decoder(OUTPUT_DIM,
                          HID_DIM,
                          DEC_LAYERS,
                          DEC_HEADS,
                          DEC_PF_DIM,
                          DEC_DROPOUT,
                          DEVICE)

            model = Transformer(enc, dec, DEVICE, MODEL_DIR, MODEL_FILE_NAME).to(DEVICE)

            trainer = Trainer(model, preprocessor._tokenizer_.vocab['pad'], LEARNING_RATE, clip=CLIP)
            train_loss, test_loss = trainer(TR_SET, TS_SET, epochs=epochs)
            translator = pipeline.create_translator(model, preprocessor)
            bleu_results = pipeline.bleu_evaluation(translator, dataset.data)

            gs_dict_iter = {
                "hyperparams": {
                    "PRETRAINED": PRETRAINED,
                    "HID_DIM": HID_DIM,
                    "ENC_LAYERS": ENC_LAYERS,
                    "DEC_LAYERS": DEC_LAYERS,
                    "ENC_PF_DIM": ENC_PF_DIM,
                    "DEC_PF_DIM": DEC_PF_DIM,
                    "LEARNING_RATE": LEARNING_RATE,
                    "CLIP": CLIP
                },
                "loss": {
                    "train": train_loss,
                    "test": test_loss,
                    "zeroshot_train": trainer.evaluate_loss(ZS_TR_SET),
                    "zeroshot_test": trainer.evaluate_loss(ZS_TS_SET)
                },
                "accuracy": {
                    "train": trainer.evaluate_metric(TR_SET),
                    "test": trainer.evaluate_metric(TS_SET),
                    "zeroshot_train": trainer.evaluate_metric(ZS_TR_SET),
                    "zeroshot_test": trainer.evaluate_metric(ZS_TS_SET)
                },
                "metric": bleu_results
            }
            with open(self.gs_file, 'w') as fp:
                self.gs_dict.append(gs_dict_iter)
                json.dump(self.gs_dict, fp)


if __name__ == "__main__":
    gs = GridSearch(Hyperparameters())
    gs.train(10)
