from pathlib import Path
import matplotlib.pyplot as plt

class PlotHandler:

    def __init__(self, file_dir: Path, file_name: str):
        self.loss_train_list = []
        self.loss_test_list = []
        self.loss_zero_shot_list = []
        self.file_dir = file_dir
        self.file_name = file_name

    def model_callback(self, loss_train, loss_val, zero_shot_loss, time):
        self.loss_train_list.append(loss_train)
        self.loss_test_list.append(loss_val)
        self.loss_zero_shot_list.append(zero_shot_loss)

    def save_plot(self, show_plot = False):
        plt.plot(list(range(1, len(self.loss_train_list) + 1)), self.loss_train_list, label="loss_train")
        plt.plot(list(range(1, len(self.loss_test_list) + 1)), self.loss_test_list, label="loss_test")
        plt.plot(list(range(1, len(self.loss_zero_shot_list) + 1)), self.loss_zero_shot_list, label="loss_zero_shot")
        plt.legend()
        plt.grid(True)
        if not self.file_dir.exists():
            self.file_dir.mkdir()
        plt.savefig(self.file_dir / self.file_name)
        if show_plot:
            plt.show()