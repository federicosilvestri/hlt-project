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
from preprocessing import Preprocessor, PreprocessSerializer
from utils.plot_handler import PlotHandler

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
        preprocessor_serializer = PreprocessSerializer(
            file_name=PREPROCESSOR_FILE_NAME, file_dir=PREPROCESSOR_DIR
        )

        if not preprocessor_serializer.exists():
            logging.info("Preprocessing file not found, executing preprocessing...")
            preprocessor = Preprocessor(dataset=dataset, max_length=MAX_LENGTH, chunks=CHUNKS, limit=limit)
            # executing preprocessing
            preprocessor.execute()

            # saving
            logging.info("Saving preprocessor into file")
            preprocessor_serializer.serialize(preprocessor)
        else:
            logging.info("Loading preprocessor from file")
            preprocessor = preprocessor_serializer.load()
        return preprocessor

    def holdout(self, data, thresh_perd=HOLDOUT_VALID_FRACTION):
        #
        # Hold out
        #
        logging.info("Splitting dataset in hold out way")
        train_data_size = len(data)
        threshold = int(train_data_size - train_data_size * thresh_perd)
        TR_SET = data[:threshold]
        TS_SET = data[threshold:]
        return TR_SET, TS_SET

    def model_creation(self, preprocessor):
        #
        # Model creaton
        #
        logging.info("Transformer creation")
        INPUT_DIM = len(preprocessor._tokenizer_)
        OUTPUT_DIM = len(preprocessor._tokenizer_)

        if PRETRAINED:
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

        return Transformer(enc, dec, DEVICE, MODEL_DIR, MODEL_FILE_NAME).to(DEVICE)

    def train_model(self, model, TRG_INDEX_PAD, TR_SET, TS_SET, epochs=EPOCHS, clip=CLIP, learning_rate=LEARNING_RATE,
                    callbacks=[]):
        #
        # Model training
        #
        trainer = Trainer(model, TRG_INDEX_PAD, learning_rate=learning_rate, batch_size=BATCH_SIZE, clip=clip,
                          device=DEVICE, limit_eval=LIMIT_EVAL)
        logging.info("Start model training")
        trainer(TR_SET, TS_SET, epochs=epochs, callbacks=callbacks)
        logging.info("End model training")

    def create_translator(self, model, preprocessor, chunks=CHUNKS):
        return TransformerTranslator(model, preprocessor._tokenizer_,
                                     preprocessor._tokenizer_, MAX_LENGTH, chunks, DEVICE, limit_bleu=LIMIT_BLEU)

    def translate(self, translator, dataset, limit=6):
        #
        # Translation of some sentences
        #
        ZERO_SHOT_SET = [(f"[2it] {key}", value['it']) for key, value in list(dataset.data.items())[:limit]]
        ZS_TRAIN, ZS_TEST = self.holdout(ZERO_SHOT_SET, thresh_perd=0.5)

        logging.info("Printing some sentances translation in zero-shot train and test way")
        for key, value in ZS_TRAIN + ZS_TEST:
            pred = translator(key)
            print(f'SRC: {key}')
            print(f'OUT: {value}')
            print(f'PRED: {pred}')
            print()

    def bleu_evaluation(self, translator, dataset, limit=None):
        ZERO_SHOT_SET = [(f"[2it] {key}", value['it']) for key, value in list(dataset.items())]
        DATA_SET = [(f"[2fr] {key}", value['fr']) for key, value in list(dataset.items())] + \
                   [(f"[2de] {key}", value['de']) for key, value in list(dataset.items())] + \
                   [(f"[2es] {key}", value['es']) for key, value in list(dataset.items())] + \
                   [(f"[2it] {value['fr']}", value['it']) for key, value in list(dataset.items())] + \
                   [(f"[2it] {value['de']}", value['it']) for key, value in list(dataset.items())] + \
                   [(f"[2it] {value['es']}", value['it']) for key, value in list(dataset.items())]
        DT_TRAIN, DT_TEST = self.holdout(DATA_SET)
        ZS_TRAIN, ZS_TEST = self.holdout(ZERO_SHOT_SET)
        if limit is not None:
            DT_TRAIN = DT_TRAIN[:limit]
            DT_TEST = DT_TEST[:limit]
            ZS_TRAIN = ZS_TRAIN[:limit]
            ZS_TEST = ZS_TEST[:limit]
        test_sets = [
            ("zeroshot_test", ZS_TEST),
            ("zeroshot_train", ZS_TRAIN),
            ("dataset_test", DT_TEST),
            ("dataset_train", DT_TRAIN),
        ]
        #
        # Evaluation of model using BLEU and sacreBLEU
        #
        results = {}
        for label, test_set in test_sets:
            logging.info(f"Translated set {label} creation")
            translated_set = translator.create_translatedset(test_set)

            logging.info("BLEU score computation")
            bleu_score = translated_set.bleu()
            print(f'BLEU score {label} = {bleu_score * 100:.2f}%')

            logging.info("sacreBLEU score computation")
            sacre_bleu_score = translated_set.sacre_bleu()

            print(f'sacreBLEU score {label} = {sacre_bleu_score * 100:.2f}%')

            results[label] = {
                "BLEU": bleu_score,
                "sacreBLEU": sacre_bleu_score
            }
        return results
