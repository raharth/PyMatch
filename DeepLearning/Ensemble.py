import torch


class Ensemble:

    def __init__(self, trainer_factory, n_model, trainer_args={}):
        self.trainers = []
        for i in range(n_model):
            trainer_args['name'] = '{}'.format(i)
            self.trainers.append(trainer_factory(**trainer_args))
        self.epochs_run = 0

        self.losses = []

    def predict(self, x, device='cpu'):
        y_preds = [trainer.predict(x, device, prob=True) for trainer in self.trainers]
        y_preds = torch.stack(y_preds)
        y_pred_mean = y_preds.mean(dim=0)
        y_pred_std = y_preds.std(dim=0)
        return y_pred_mean, y_pred_std

    def train(self, epochs, device, checkpoint_int=10, validation_int=10, restore_early_stopping=False, verbose=1):
        for epoch in range(0, epochs, validation_int):
            if verbose == 1:
                print('Ensemble epochs {}-{}'.format(epoch, epoch + validation_int - 1))
            for idx, trainer in enumerate(self.trainers):
                if verbose == 1:
                    print('Trainer {}'.format(idx))
                trainer.train(epochs=validation_int, device=device, checkpoint_int=checkpoint_int,
                              validation_int=validation_int, restore_early_stopping=restore_early_stopping)

    def dump_checkpoint(self, path=None, tag='checkpoint'):
        for trainer in self.trainers:
            trainer.dump_checkpoint(path=path, tag=tag)

    def load_checkpoint(self, path=None, tag='checkpoint'):
        for trainer in self.trainers:
            trainer.load_checkpoint(path=path, tag=tag)
