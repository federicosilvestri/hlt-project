from pipeline import Pipeline
from trainer.trainer_callbacks import print_epoch_loss_accuracy
from utils.plot_handler import PlotHandlerFactory
from config import *

pipeline = Pipeline()

dataset = pipeline.dataset_load()
structured_dataset = pipeline.preprocess(dataset)
print(f"Structured dataset sizes\n{structured_dataset.sizes()}")

model = pipeline.model_creation()
translator = pipeline.create_translator(model)

print_callback = print_epoch_loss_accuracy(structured_dataset)
plot_handler_factory = PlotHandlerFactory(PLOTS_DIR, structured_dataset)

pipeline.train_model(model, TOKENIZER.vocab['pad'], structured_dataset, callbacks=[
    structured_dataset.model_callback(translator),
    print_callback,
    plot_handler_factory.create_celoss_plot().model_callback,
    plot_handler_factory.create_accuracy_plot().model_callback,
    plot_handler_factory.create_bleu_plot().model_callback,
    plot_handler_factory.create_sacrebleu_plot().model_callback,
])
plot_handler_factory.save_all()

pipeline.translate(translator, structured_dataset)
