import random
import math
from typing import Callable, List, Tuple
import torch
from torch import nn
import logging as lg
from torchmetrics import Accuracy


class Trainer:
    """This class create the instance able to train a trnsformer model for a NMT task."""

    def __init__(self, model: nn.Module, learning_rate: float = 0.0005, batch_size: int = 32,
                 clip: int = 1, device='cpu', limit_eval=None, ) -> None:
        """Trainer constructor

        Args:
            model (nn.Module): Transformer model that will be trained.
            learning_rate (float, optional): Learning rate used from Adam optimizer.
            batch_size (int): Batch size to create minibatches.
            clip (int, optional): Parameter used to clip weights norm each minibatch training.
        """
        self.model = model
        self.clip = clip
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy().to(device)
        self.device = device
        self.limit_eval = limit_eval

    def __create_batches(self, train_set: List[Tuple[torch.tensor, torch.tensor]]) -> List[
        Tuple[torch.tensor, torch.tensor]]:
        """Method able to create batches of a train set.

        Args:
            train_set (List[Tuple[torch.tensor, torch.tensor]]): Original training set.

        Returns:
            List[Tuple[torch.tensor, torch.tensor]]: Minibatch training set.
        """
        num_batches = math.ceil(len(train_set) / self.batch_size)
        minibatch_tr = []
        for num_batch in range(num_batches):
            stop_index = (num_batch + 1) * self.batch_size
            stop_index = stop_index if len(
                train_set) > stop_index else len(train_set)
            batch_train_data = train_set[num_batch *
                                         self.batch_size: stop_index]
            src, trg = zip(*batch_train_data)
            minibatch_tr.append((torch.cat(src, 0), torch.cat(trg, 0)))
        return minibatch_tr

    def train(self, train_set: List[Tuple[torch.tensor, torch.tensor]]) -> float:
        """Method used to train one single epoch.

        Args:
            train_set (List[Tuple[torch.tensor, torch.tensor]])): Train set.

        Return:
            float: Training loss related to the current epoch.
        """
        self.model.train()
        epoch_loss = 0
        for src, trg in train_set:
            self.optimizer.zero_grad()
            output, _ = self.model(src, trg[:, :-1])
            # output = [batch size, trg len - 1, output dim]
            # trg = [batch size, trg len]
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim).to(self.device)
            trg = trg[:, 1:].contiguous().view(-1).to(self.device)
            # output = [batch size * trg len - 1, output dim]
            # trg = [batch size * trg len - 1]
            loss = self.criterion(output, trg)
            loss.backward()
            if self.clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / len(train_set)

    def evaluate_loss(self, test_set: List[Tuple[torch.tensor, torch.tensor]]) -> float:
        """Method used to evaluate loss on one single epoch.

          Args:
              test_set (List[Tuple[torch.tensor, torch.tensor]])): Test set.

          Return:
              float: Test loss related to the current epoch.
        """
        self.model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for src, trg in test_set[:self.limit_eval]:
                output, _ = self.model(src, trg[:, :-1])
                # output = [batch size, trg len - 1, output dim]
                # trg = [batch size, trg len]
                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim).to(self.device)
                trg = trg[:, 1:].contiguous().view(-1).to(self.device)
                # output = [batch size * trg len - 1, output dim]
                # trg = [batch size * trg len - 1]
                loss = self.criterion(output, trg)
                epoch_loss += loss.item()
        return epoch_loss / len(test_set)

    def evaluate_metric(self, test_set: List[Tuple[torch.tensor, torch.tensor]]) -> float:
        """Method used to evaluate metric on one single epoch.

          Args:
              test_set (List[Tuple[torch.tensor, torch.tensor]])): Test set.

          Return:
              float: Test accuracy related to the current epoch.
        """
        self.model.eval()
        epoch_acc = 0
        with torch.no_grad():
            for src, trg in test_set[:self.limit_eval]:
                output, _ = self.model(src, trg[:, :-1])
                # output = [batch size, trg len - 1, output dim]
                # trg = [batch size, trg len]
                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim).argmax(-1).to(self.device)
                trg = trg[:, 1:].contiguous().view(-1).to(self.device)
                # output = [batch size * trg len - 1, output dim]
                # trg = [batch size * trg len - 1]
                accuracy = self.accuracy(output, trg)
                epoch_acc += accuracy.item()
        return epoch_acc / len(test_set)

    def __call__(self, train_set: List[Tuple[torch.tensor, torch.tensor]],
                 epochs: int = 10, callbacks: List[Callable] = [],
                 save_model=True) -> None:
        """Main method of the class able to train the model.

        Args:
            train_set (List[Tuple[torch.tensor, torch.tensor]])): Training set.
            epochs (int, optional): Number of time that the train process is repeated.
            callbacks (List[Callable], optional): List of callbacks called after each epoch.
        """
        lg.info("start training")
        for epoch in range(epochs):
            lg.info(f"Start epoch {epoch}")
            random.shuffle(train_set)
            mb_train_set = self.__create_batches(train_set)
            lg.info(f"Start train epoch {epoch}")
            train_loss = self.train(mb_train_set)
            lg.info(f"Stop train epoch {epoch}")
            if save_model:
                self.model.save_transformer()
            lg.info(f"Start callbacks epoch {epoch}")
            for i, callback in enumerate(callbacks):
                lg.info(f"Start callback {i} epoch {epoch}")
                callback(self, epoch, train_loss)
                lg.info(f"Stop callback {i} epoch {epoch}")
            lg.info(f"Stop callbacks epoch {epoch}")
        lg.info(f"Stop epoch {epoch}")
        return train_loss
