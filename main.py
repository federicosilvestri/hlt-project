from pipeline import Pipeline
from tokenizer import Tokenizer
from transformers import XLMRobertaTokenizer
from trainer.trainer_callbacks import print_epoch_loss_accuracy
from utils.plot_handler import PlotHandlerFactory
from config import *

print(f"DEVICE '{DEVICE}' with max N_DEGREE {N_DEGREE}")

tokenizer = Tokenizer(XLMRobertaTokenizer.from_pretrained('xlm-roberta-base'), device=DEVICE)
pipeline = Pipeline(tokenizer)

dataset = pipeline.dataset_load()


structured_dataset = pipeline.preprocess(dataset)
print(f"Structured dataset sizes\n{structured_dataset.sizes()}")

model = pipeline.model_creation(
    # None (for personal model), 'bert', 'distilbert', 'mt5'
    type='xmlroberta',
    # only with pretrained:
    #       'bert-base-multilingual-uncased'            <- bert
    #       'distilbert-base-multilingual-uncased'      <- distilbert
    #       'xlm-roberta-base'                          <- xlmroberta
    #       'google/mt5-small'                          <- mt5
    pretrained_type=None,
    enc_layers=ENC_LAYERS,
    enc_heads=ENC_HEADS,
    # personal:                             HID_DIM
    # bert-base-multilingual-uncased:         768
    # distilbert-base-multilingual-uncased:   768
    # google/mt5-small:                     512
    hid_dim=HID_DIM,
    dec_layers=DEC_LAYERS,
    dec_heads=DEC_HEADS
)
translator = pipeline.create_translator(model)

print_callback = print_epoch_loss_accuracy(structured_dataset)
plot_handler_factory = PlotHandlerFactory(PLOTS_DIR, structured_dataset)

trainer = pipeline.train_model(model, structured_dataset, epochs=1, callbacks=[
    structured_dataset.model_callback(translator),
    print_callback,
    # plot_handler_factory.create_celoss_plot().model_callback,
    # plot_handler_factory.create_accuracy_plot().model_callback,
    # plot_handler_factory.create_bleu_plot().model_callback,
    # plot_handler_factory.create_sacrebleu_plot().model_callback,
])
plot_handler_factory.save_all()

pipeline.translate(translator, structured_dataset)

print(structured_dataset.to_dict(trainer, translator))
