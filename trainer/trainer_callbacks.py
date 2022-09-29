def print_epoch_loss_accuracy(train_set, test_set, zs_train_set, zs_test_set):
    print_loss_acc = lambda name, loss, acc: f'\t{name}\t|\tloss:\t{loss:.3f}\t|\taccuracy:\t{acc:.3f}'
    def callback(trainer, epoch, train_loss, test_loss, timer):
        epoch_mins, epoch_secs = timer
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(print_loss_acc('Dataset TR-set', train_loss, trainer.evaluate_metric(train_set)))
        print(print_loss_acc('Dataset TS-set', test_loss, trainer.evaluate_metric(test_set)))
        print(print_loss_acc('Zero-shot TR-set', trainer.evaluate_loss(zs_train_set), trainer.evaluate_metric(zs_train_set)))
        print(print_loss_acc('Zero-shot TS-set', trainer.evaluate_loss(zs_test_set), trainer.evaluate_metric(zs_test_set)))
    return callback