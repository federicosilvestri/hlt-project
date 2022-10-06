from pipeline import Pipeline
from trainer.trainer_callbacks import print_epoch_loss_accuracy
from utils.plot_handler import PlotHandlerFactory
from config import *

pipeline = Pipeline()

dataset = pipeline.dataset_load()
preprocessor = pipeline.preprocess(dataset)

limit_labels = 10
dataset_labels = list(dataset.data.items())[:limit_labels]
DATA_SET = [(f"[2fr] {key}", value['fr']) for key, value in dataset_labels] + \
           [(f"[2de] {key}", value['de']) for key, value in dataset_labels] + \
           [(f"[2es] {key}", value['es']) for key, value in dataset_labels] + \
           [(f"[2it] {value['fr']}", value['it']) for key, value in dataset_labels] + \
           [(f"[2it] {value['de']}", value['it']) for key, value in dataset_labels] + \
           [(f"[2it] {value['es']}", value['it']) for key, value in dataset_labels]
ZEROSHOT_SET = [(f"[2it] {key}", value['it']) for key, value in dataset_labels]
TR_SET_LABEL, TS_SET_LABEL = pipeline.holdout(DATA_SET)
ZS_TR_SET_LABEL, ZS_TS_SET_LABEL = pipeline.holdout(ZEROSHOT_SET)
TR_SET, TS_SET = pipeline.holdout(preprocessor.trainable_data)
ZS_TR_SET, ZS_TS_SET = pipeline.holdout(preprocessor.zeroshot_data)

model = pipeline.model_creation(preprocessor)
translator = pipeline.create_translator(model, preprocessor)

print_callback = print_epoch_loss_accuracy(TR_SET, TS_SET, ZS_TR_SET, ZS_TS_SET)
plot_handler_factory = PlotHandlerFactory(PLOTS_DIR)

pipeline.train_model(model, preprocessor._tokenizer_.vocab['pad'], TR_SET, TS_SET, callbacks=[
    print_callback,
    plot_handler_factory.create_celoss_plot(ZS_TR_SET, ZS_TS_SET).model_callback,
    plot_handler_factory.create_accuracy_plot(TR_SET, TS_SET, ZS_TR_SET, ZS_TS_SET).model_callback,
    plot_handler_factory.create_bleu_plot(translator, TR_SET_LABEL, TS_SET_LABEL, ZS_TR_SET_LABEL, ZS_TS_SET_LABEL).model_callback,
    plot_handler_factory.create_sacrebleu_plot(translator, TR_SET_LABEL, TS_SET_LABEL, ZS_TR_SET_LABEL, ZS_TS_SET_LABEL).model_callback,
])
plot_handler_factory.save_all()

pipeline.translate(translator, dataset)
pipeline.bleu_evaluation(translator, dataset.data, limit=100)
