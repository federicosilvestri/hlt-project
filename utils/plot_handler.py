from pathlib import Path
from typing import List

import matplotlib.pyplot as plt

from data.structured_dataset import StructuredDataset
from translate.transformer_translator import TransformerTranslator


class PlotHandler:

    def __init__(self, name, file_dir: Path, file_name: str, callbacks: []):
        self.train_list = []
        self.test_list = []
        self.zero_shot_train_list = []
        self.zero_shot_test_list = []
        self.callbacks = callbacks
        self.name = name
        self.file_dir = file_dir
        self.file_name = file_name

    def model_callback(self, trainer, epoch, loss_train, loss_val, time):
        self.train_list.append(self.callbacks[0](trainer, epoch, loss_train, loss_val))
        self.test_list.append(self.callbacks[1](trainer, epoch, loss_train, loss_val))
        self.zero_shot_train_list.append(self.callbacks[2](trainer, epoch, loss_train, loss_val))
        self.zero_shot_test_list.append(self.callbacks[3](trainer, epoch, loss_train, loss_val))

    def save_plot(self):
        plt.figure()
        plt.plot(list(range(1, len(self.train_list) + 1)), self.train_list, label=f"{self.name}_train")
        plt.plot(list(range(1, len(self.test_list) + 1)), self.test_list, label=f"{self.name}_test")
        plt.plot(list(range(1, len(self.zero_shot_train_list) + 1)), self.zero_shot_train_list, label=f"{self.name}_zeroshot_train")
        plt.plot(list(range(1, len(self.zero_shot_test_list) + 1)), self.zero_shot_test_list, label=f"{self.name}_zeroshot_test")
        plt.legend()
        plt.grid(True)
        if not self.file_dir.exists():
            self.file_dir.mkdir()
        plt.savefig(self.file_dir / self.file_name)



class PlotHandlerFactory:
    def __init__(self, plots_dir, structured_dataset: StructuredDataset):
        self.plots_dir = plots_dir
        self.plot_handlers: List[PlotHandler] = []
        self.structured_dataset = structured_dataset
    def create_celoss_plot(self):
        plot_handler = PlotHandler('celoss', self.plots_dir, "celoss_plot.png", [
            lambda trainer, epoch, loss_train, loss_val: loss_train,
            lambda trainer, epoch, loss_train, loss_val: loss_val,
            lambda trainer, epoch, loss_train, loss_val: self.structured_dataset.zeroshotset.train.loss,
            lambda trainer, epoch, loss_train, loss_val: self.structured_dataset.zeroshotset.test.loss,
        ])
        self.plot_handlers.append(plot_handler)
        return plot_handler

    def create_accuracy_plot(self):
        plot_handler = PlotHandler('accuracy', self.plots_dir, "accuracy_plot.png", [
            lambda trainer, epoch, loss_train, loss_val: self.structured_dataset.baseset.train.accuracy,
            lambda trainer, epoch, loss_train, loss_val: self.structured_dataset.baseset.test.accuracy,
            lambda trainer, epoch, loss_train, loss_val: self.structured_dataset.zeroshotset.train.accuracy,
            lambda trainer, epoch, loss_train, loss_val: self.structured_dataset.zeroshotset.test.accuracy,
        ])
        self.plot_handlers.append(plot_handler)
        return plot_handler


    def create_bleu_plot(self):
        plot_handler = PlotHandler('bleu', self.plots_dir, "bleu_plot.png", [
            lambda trainer, epoch, loss_train, loss_val: self.structured_dataset.baseset.train.bleu,
            lambda trainer, epoch, loss_train, loss_val: self.structured_dataset.baseset.test.bleu,
            lambda trainer, epoch, loss_train, loss_val: self.structured_dataset.zeroshotset.train.bleu,
            lambda trainer, epoch, loss_train, loss_val: self.structured_dataset.zeroshotset.test.bleu,
        ])
        self.plot_handlers.append(plot_handler)
        return plot_handler

    def create_sacrebleu_plot(self):
        plot_handler = PlotHandler('sacrebleu', self.plots_dir, "sacrebleu_plot.png", [
            lambda trainer, epoch, loss_train, loss_val: self.structured_dataset.baseset.train.sacrebleu,
            lambda trainer, epoch, loss_train, loss_val: self.structured_dataset.baseset.test.sacrebleu,
            lambda trainer, epoch, loss_train, loss_val: self.structured_dataset.zeroshotset.train.sacrebleu,
            lambda trainer, epoch, loss_train, loss_val: self.structured_dataset.zeroshotset.test.sacrebleu,
        ])
        self.plot_handlers.append(plot_handler)
        return plot_handler

    def save_all(self):
        for plot_handler in self.plot_handlers:
            plot_handler.save_plot()