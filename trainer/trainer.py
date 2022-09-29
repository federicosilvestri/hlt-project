import random
import math
import time
from typing import Callable, List, Tuple
import torch
from torch import nn
import logging as lg
from torchmetrics import Accuracy


class Trainer:
    """This class create the instance able to train a trnsformer model for a NMT task."""

    def __init__(self, model: nn.Module, trg_pad_idx: int, learning_rate: float = 0.0005, batch_size: int = 32, clip: int = 1) -> None:
        """Trainer constructor

        Args:
            model (nn.Module): Transformer model that will be trained.
            trg_pad_idx (int): Index that represent padding tag on decoder tokenizer.
            learning_rate (float, optional): Learning rate used from Adam optimizer.
            batch_size (int): Batch size to create minibatches.
            clip (int, optional): Parameter used to clip weights norm each minibatch training.
        """
        self.model = model
        self.clip = clip
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)
        self.accuracy = Accuracy()

    def __epoch_time(self, start_time: float, end_time: float) -> Tuple[int, int]:
        """Method able to extract minutes and seconds from a start and an end time.

        Args:
            start_time (float): Start time of a counter.
            end_time (float): End time of a counter.

        Returns:
            Tuple[int, int]: Duration in minutes and seconds.
        """
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def __create_batches(self, train_set: List[Tuple[torch.tensor, torch.tensor]]) -> List[Tuple[torch.tensor, torch.tensor]]:
        """Method able to create batches of a train set.

        Args:
            train_set (List[Tuple[torch.tensor, torch.tensor]]): Original training set.

        Returns:
            List[Tuple[torch.tensor, torch.tensor]]: Minibatch training set.
        """
        num_batches = math.ceil(len(train_set) / self.batch_size)
        minibatch_tr = []
        for num_batch in range(num_batches):
            stop_index = (num_batch+1) * self.batch_size
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
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
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
            for src, trg in test_set:
                output, _ = self.model(src, trg[:, :-1])
                # output = [batch size, trg len - 1, output dim]
                # trg = [batch size, trg len]
                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim)
                trg = trg[:, 1:].contiguous().view(-1)
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
              float: Test loss related to the current epoch.
        """
        self.model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for src, trg in test_set:
                output, _ = self.model(src, trg[:, :-1])
                # output = [batch size, trg len - 1, output dim]
                # trg = [batch size, trg len]
                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim)
                trg = trg[:, 1:].contiguous().view(-1)
                # output = [batch size * trg len - 1, output dim]
                # trg = [batch size * trg len - 1]
                loss = self.accuracy(output.argmax(-1), trg)
                epoch_loss += loss.item()
        return epoch_loss / len(test_set)

    def __call__(self, train_set: List[Tuple[torch.tensor, torch.tensor]],
                 test_set: List[Tuple[torch.tensor, torch.tensor]],
                 epochs: int = 10, callbacks: List[Callable] = []) -> None:
        """Main method of the class able to train the model.

        Args:
            train_set (List[Tuple[torch.tensor, torch.tensor]])): Training set.
            test_set (List[Tuple[torch.tensor, torch.tensor]])): Test set.
            epochs (int, optional): Number of time that the train process is repeated.
            callbacks (List[Callable], optional): List of callbacks called after each epoch.
        """
        lg.info("Start training")
        for epoch in range(epochs):
            start_time = time.time()
            random.shuffle(train_set)
            mb_train_set = self.__create_batches(train_set)
            train_loss = self.train(mb_train_set)
            test_loss = self.evaluate_loss(test_set)
            end_time = time.time()
            self.model.save_transformer()
            epoch_mins, epoch_secs = self.__epoch_time(start_time, end_time)
            for callback in callbacks:
                callback(self, epoch, train_loss, test_loss, (epoch_mins, epoch_secs))
        return train_loss, test_loss
