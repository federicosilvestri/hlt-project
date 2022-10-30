from pipeline import Pipeline
from tokenizer import Tokenizer
from transformers import MT5Tokenizer
from trainer.trainer_callbacks import print_epoch_loss_accuracy
from utils.plot_handler import PlotHandlerFactory
from config import *

print(f"DEVICE '{DEVICE}' with max N_DEGREE {N_DEGREE}")

tokenizer = Tokenizer(MT5Tokenizer.from_pretrained('google/mt5-small'), device=DEVICE)
pipeline = Pipeline(tokenizer)

dataset = pipeline.dataset_load()


structured_dataset = pipeline.preprocess(dataset)
print(f"Structured dataset sizes\n{structured_dataset.sizes()}")

model = pipeline.model_creation(
    # None (for personal model), 'bert', 'mt5'
    type='mt5',
    # only with pretrained:
    #       'bert-base-multilingual-uncased'            <- bert
    #       'distilbert-base-multilingual-uncased'      <- bert
    #       'google/mt5-small'                          <- mt5
    pretrained_type='google/mt5-small',
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
