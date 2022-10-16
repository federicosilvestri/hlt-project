from data.structured_dataset import StructuredDataset


def print_epoch_loss_accuracy(structured_dataset: StructuredDataset):
    def print_loss_acc(name, loss, acc=None, bleu=None, sacrebleu=None):
        return f'\t{name}\t|\tloss:\t{loss:.3f}' + (f'\t|\taccuracy:\t{acc:.3f}' if acc is not None else '') + (
            f'\t|\tbleu:\t{bleu:.3f}' if bleu is not None else '') + (
                   f'\t|\tsacrebleu:\t{sacrebleu:.3f}' if sacrebleu is not None else '')

    def callback(trainer, epoch, train_loss):
        print(f'Epoch: {epoch + 1:02}')
        print(print_loss_acc('Dataset  TR-set', structured_dataset.baseset.train.loss,
                             structured_dataset.baseset.train.accuracy,
                             structured_dataset.baseset.train.bleu,
                             structured_dataset.baseset.train.sacrebleu))
        if structured_dataset.baseset.test.loss is not None:
            print(print_loss_acc('Dataset  TS-set', structured_dataset.baseset.test.loss,
                                 structured_dataset.baseset.test.accuracy,
                                 structured_dataset.baseset.test.bleu,
                                 structured_dataset.baseset.test.sacrebleu))
        if structured_dataset.zeroshotset.train.loss is not None:
            print(print_loss_acc('Zeroshot TR-set', structured_dataset.zeroshotset.train.loss,
                                 structured_dataset.zeroshotset.train.accuracy,
                                 structured_dataset.zeroshotset.train.bleu,
                                 structured_dataset.zeroshotset.train.sacrebleu))
        if structured_dataset.zeroshotset.test.loss is not None:
            print(print_loss_acc('Zeroshot TS-set', structured_dataset.zeroshotset.test.loss,
                                 structured_dataset.zeroshotset.test.accuracy,
                                 structured_dataset.zeroshotset.test.bleu,
                                 structured_dataset.zeroshotset.test.sacrebleu))

    return callback
