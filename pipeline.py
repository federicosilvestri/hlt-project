from data.serializer import SDSerializer
from data.structured_dataset import StructuredDataset
from model.bert_encoder import BERTEncoder
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

    def preprocess(self, dataset, limit=None):
        #
        # Execute preprocessing
        #
        serializer = SDSerializer(
            file_name=PREPROCESSOR_FILE_NAME, file_dir=PREPROCESSOR_DIR
        )

        if not serializer.exists():
            logging.info("Preprocessing file not found, executing preprocessing...")
            preprocessor = Preprocessor(dataset=dataset, tokenizer=TOKENIZER, max_length=MAX_LENGTH, chunks=CHUNKS,
                                        limit=limit)
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

    def model_creation(self):
        #
        # Model creaton
        #
        logging.info("Transformer creation")
        INPUT_DIM = VOCAB_SIZE
        OUTPUT_DIM = VOCAB_SIZE

        if PRETRAINED_TYPE is not None:
            enc = BERTEncoder(HID_DIM, ENC_HEADS, VOCAB_SIZE, DEVICE, type=PRETRAINED_TYPE)
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

        return Transformer(enc, dec, DEVICE, MODEL_DIR, MODEL_FILE_NAME).to(DEVICE)

    def train_model(self, model, TRG_INDEX_PAD, structured_dataset: StructuredDataset, epochs=EPOCHS, clip=CLIP,
                    learning_rate=LEARNING_RATE,
                    callbacks=[]):
        #
        # Model training
        #
        trainer = Trainer(model, TRG_INDEX_PAD, learning_rate=learning_rate, batch_size=BATCH_SIZE, clip=clip,
                          device=DEVICE, limit_eval=LIMIT_EVAL)
        logging.info("Start model training")
        trainer(structured_dataset.baseset.train.tokens_id, structured_dataset.baseset.test.tokens_id, epochs=epochs,
                callbacks=callbacks)
        logging.info("End model training")

    def create_translator(self, model, tokenizer=TOKENIZER, chunks=CHUNKS):
        return TransformerTranslator(model, tokenizer, tokenizer, MAX_LENGTH, chunks, DEVICE, limit_bleu=LIMIT_BLEU)

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
