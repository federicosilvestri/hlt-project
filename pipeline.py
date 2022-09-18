from model.bert_encoder import BERTEncoder
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
        return Dataset(dataset_downloader.downloaded_file)

    def preprocess(self, dataset):
        #
        # Execute preprocessing
        #
        preprocessor_serializer = PreprocessSerializer(
            file_name=PREPROCESSOR_FILE_NAME, file_dir=PREPROCESSOR_DIR
        )

        if not preprocessor_serializer.exists():
            logging.info("Preprocessing file not found, executing preprocessing...")
            preprocessor = Preprocessor(dataset=dataset, max_length=100, limit=None)
            # executing preprocessing
            preprocessor.execute()

            # saving
            logging.info("Saving preprocessor into file")
            preprocessor_serializer.serialize(preprocessor)
        else:
            logging.info("Loading preprocessor from file")
            preprocessor = preprocessor_serializer.load()
        return preprocessor

    def holdout(self, preprocessor):
        #
        # Hold out
        #
        logging.info("Splitting dataset in hold out way")
        train_data_size = len(preprocessor.trainable_data)
        threshold = int(train_data_size - train_data_size * HOLDOUT_VALID_FRACTION)
        TR_SET = preprocessor.trainable_data[:threshold]
        TS_SET = preprocessor.trainable_data[threshold:]
        zs_train_data_size = len(preprocessor.zeroshot_data)
        threshold = int(zs_train_data_size - zs_train_data_size * HOLDOUT_VALID_FRACTION)
        ZS_TR_SET = preprocessor.trainable_data[:threshold]
        ZS_TS_SET = preprocessor.trainable_data[threshold:]
        return TR_SET, TS_SET, ZS_TR_SET, ZS_TS_SET

    def model_creation(self, preprocessor):
        #
        # Model creaton
        #
        logging.info("Transformer creation")
        SRC_PAD_IDX = preprocessor._pad_index_
        TRG_PAD_IDX = preprocessor._pad_index_
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

        return Transformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, DEVICE, MODEL_DIR, MODEL_FILE_NAME).to(DEVICE)

    def train_model(self, model, TRG_PAD_IDX, TR_SET, TS_SET, ZS_TR_SET, ZS_TS_SET):
        #
        # Setup plot handler
        #
        plot_handler = PlotHandler(PLOTS_DIR, LOSS_PLOT_FILE_NAME)

        #
        # Model training
        #
        trainer = Trainer(model, trg_pad_idx=TRG_PAD_IDX, batch_size=BATCH_SIZE)
        logging.info("Start model training")
        trainer(TR_SET, TS_SET, ZS_TR_SET, ZS_TS_SET, epochs=EPOCHS, callbacks=[plot_handler.model_callback])
        logging.info("End model training")

        #
        # Save plot locally
        #
        plot_handler.save_plot()

    def translate(self, model, dataset, preprocessor):
        #
        # Translation of some sentences
        #
        ZERO_SHOT_SET = [(f"[2it] {key}", value['it']) for key, value in list(dataset.data.items())]
        translator = TransformerTranslator(model, preprocessor._tokenizer_,
                                           preprocessor._tokenizer_, MAX_LENGTH, DEVICE)

        logging.info("Printing first 5 sentance translation in zero-shot way")
        for key, value in ZERO_SHOT_SET[:5]:
            pred = translator(key)
            print(f'SRC: {key}')
            print(f'OUT: {value}')
            print(f'PRED: {pred}')
            print()

        return translator

    def bleu_evaluation(self, translator, dataset):
        ZERO_SHOT_SET = [(f"[2it] {key}", value['it']) for key, value in list(dataset.data.items())]
        #
        # Evaluation of model using BLEU and sacreBLEU
        #
        logging.info("Translated set creation")
        translated_set = translator.create_translatedset(ZERO_SHOT_SET)

        logging.info("BLEU score computation")
        bleu_score_zero_shot = translated_set.bleu()
        print(f'BLEU score zero shot = {bleu_score_zero_shot*100:.2f}%')

        logging.info("sacreBLEU score computation")
        sacre_bleu_score_zero_shot = translated_set.sacre_bleu()
        print(f'sacreBLEU score zero shot = {sacre_bleu_score_zero_shot*100:.2f}%')