from data.serializer import SDSerializer
from data.structured_dataset import StructuredDataset
from model.bert_encoder import BERTEncoder
from model.mt5_encoder import MT5Encoder
from trainer.trainer_callbacks import print_epoch_loss_accuracy
from translate.transformer_translator import TransformerTranslator
from trainer.trainer import Trainer
from model.transformer import Transformer
from model.decoder import Decoder
from model.encoder import Encoder
import logging
import sys

from config import *
from data import Dataset, DatasetDownloader
from preprocessing import Preprocessor

root = logging.getLogger()
root.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
root.addHandler(handler)


class Pipeline:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def dataset_load(self):
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
        return Dataset(dataset_downloader.downloaded_file, cut=DATASET_CUT)

    def preprocess(self, dataset, limit=None, device=DEVICE):
        #
        # Execute preprocessing
        #
        serializer = SDSerializer(
            file_name=PREPROCESSOR_FILE_NAME, file_dir=PREPROCESSOR_DIR
        )

        if not serializer.exists():
            logging.info("Preprocessing file not found, executing preprocessing...")
            preprocessor = Preprocessor(dataset=dataset, tokenizer=self.tokenizer, max_length=MAX_LENGTH,
                                        chunks=N_DEGREE,
                                        limit=limit, device=device)
            # executing preprocessing
            base_lang_config, zeroshot_lang_config = preprocessor.execute(BASE_LANG_CONFIG, ZEROSHOT_LANG_CONFIG)
            structured_dataset = StructuredDataset(base_lang_config, zeroshot_lang_config, HOLDOUT_VALID_FRACTION)

            # saving
            logging.info("Saving preprocessor into file")
            serializer.serialize(structured_dataset)
        else:
            logging.info("Loading preprocessor from file")
            structured_dataset = serializer.load()
        return structured_dataset

    def model_creation(self, type=None, pretrained_type=None, enc_layers=ENC_LAYERS, dec_layers=DEC_LAYERS,
                       hid_dim=HID_DIM,
                       enc_heads=ENC_HEADS, dec_heads=DEC_HEADS, device=DEVICE):
        #
        # Model creaton
        #
        logging.info("Transformer creation")

        if type == 'mt5':
            enc = MT5Encoder(self.tokenizer.vocab_size,
                             hid_dim,
                             enc_layers,
                             enc_heads,
                             ENC_PF_DIM,
                             ENC_DROPOUT,
                             device,
                             type=pretrained_type
                             )
        elif type == 'bert':
            enc = BERTEncoder(self.tokenizer.vocab_size,
                              hid_dim,
                              enc_layers,
                              enc_heads,
                              ENC_PF_DIM,
                              ENC_DROPOUT,
                              device,
                              type=pretrained_type
                              )
        else:
            enc = Encoder(self.tokenizer.vocab_size,
                          hid_dim,
                          enc_layers,
                          enc_heads,
                          ENC_PF_DIM,
                          ENC_DROPOUT,
                          device)

        dec = Decoder(self.tokenizer.vocab_size,
                      hid_dim,
                      dec_layers,
                      dec_heads,
                      DEC_PF_DIM,
                      DEC_DROPOUT,
                      device)

        return Transformer(enc, dec, device, MODEL_DIR, MODEL_FILE_NAME).to(device)

    def train_model(self, model, structured_dataset: StructuredDataset, epochs=EPOCHS, clip=CLIP,
                    learning_rate=LEARNING_RATE,
                    batch_size=BATCH_SIZE,
                    limit_eval=LIMIT_EVAL,
                    callbacks=[], device=DEVICE):
        #
        # Model training
        #
        trainer = Trainer(model, learning_rate=learning_rate, batch_size=batch_size, clip=clip,
                          device=device, limit_eval=limit_eval)
        logging.info("Start model training")
        trainer(structured_dataset.baseset.train.tokens_id, epochs=epochs,
                callbacks=callbacks)
        logging.info("End model training")
        return trainer

    def create_translator(self, model, chunks=N_DEGREE, device=DEVICE):
        return TransformerTranslator(model, self.tokenizer, self.tokenizer, MAX_LENGTH, chunks, device,
                                     limit_bleu=LIMIT_BLEU)

    def translate(self, translator, structured_dataset: StructuredDataset, limit=6):
        #
        # Translation of some sentences
        #
        ZS_TRAIN = structured_dataset.zeroshotset.train.labels[:limit]
        ZS_TEST = structured_dataset.zeroshotset.test.labels[:limit]

        logging.info("Printing some sentances translation in zero-shot train and test way")
        for key, value in ZS_TRAIN + ZS_TEST:
            pred = translator(key)
            print(f'SRC: {key}')
            print(f'OUT: {value}')
            print(f'PRED: {pred}')
            print()
