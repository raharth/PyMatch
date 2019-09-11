import torch
from tqdm import tqdm


class Ensemble:

    def __init__(self, trainer_factory, n_model, trainer_args={}):
        self.learners = []
        for i in range(n_model):
            t_args = dict(trainer_args)
            t_args['name'] = trainer_args['name'] + '_{}'.format(i) if 'name' in trainer_args else '{}'.format(i)
            self.learners.append(trainer_factory(**t_args))
        self.epochs_run = 0

        self.losses = []

    def predict(self, x, device='cpu'):
        # @todo pass args to the learner
        y_preds = [leaner.predict(x, device, prob=True) for leaner in self.learners]
        y_preds = torch.stack(y_preds)
        y_pred_mean = y_preds.mean(dim=0)
        y_pred_std = y_preds.std(dim=0)
        return y_pred_mean, y_pred_std

    def train(self, epochs, device, checkpoint_int=10, validation_int=10, restore_early_stopping=False, verbose=1):
        for trainer in self.learners:
            if verbose == 1:
                print('Trainer {}'.format(trainer.name))
            trainer.train(epochs=epochs, device=device, checkpoint_int=checkpoint_int,
                          validation_int=validation_int, restore_early_stopping=restore_early_stopping)

    def dump_checkpoint(self, path=None, tag='checkpoint'):
        for trainer in self.learners:
            trainer.dump_checkpoint(path=path, tag=tag)

    def load_checkpoint(self, path=None, tag='checkpoint'):
        for trainer in self.learners:
            trainer.load_checkpoint(path=path, tag=tag)

    def run_validation(self, device='cpu'):
        y_pred_learners = []
        y_true_learners = []

        for learner in self.learners:
            y_pred = []
            y_true = []
            for X, y in learner.val_loader:
                y_true += [y]
                y_pred += [learner.predict(X, device=device)]
            y_pred_learners += [torch.cat(y_pred)]
            y_true_learners += [torch.cat(y_true)]
        return y_pred_learners, y_true_learners

class BaysianEnsemble(Ensemble):

    def __init__(self, trainer_factory, n_model, trainer_args={}):
        super(BaysianEnsemble, self).__init__(trainer_factory, n_model, trainer_args=trainer_args)

    def predict(self, x, device='cpu'):
        with torch.no_grad():
            y_preds = torch.stack([trainer.model.forward(x, device=device, train=False) for trainer in self.learners])
            return y_preds.mean(dim=0).to('cpu'), y_preds.std(dim=0).to('cpu')

    def predict_class(self, x, device):
        y_pred, _ = self.predict(x, device)
        return torch.max(y_pred.data, 1)[1].data

    def predict_class_single_models(self, data_loader, device):
        for i, trainer in enumerate(self.learners):
            y_pred_list = []
            correct_pred = []

            for data, y in tqdm(data_loader):
                y_pred = trainer.predict(data, device=device, prob=False)
                y_pred_list += [y_pred]
                correct_pred += [y == y_pred.to('cpu')]

            y_pred_list = torch.cat(y_pred_list)
            correct_pred = torch.cat(correct_pred)

            print('{}: accuracy: {}'.format(i, correct_pred.float().mean()))
        # @todo no return?

    @staticmethod
    def get_confidence(y_mean, y_std):
        y_prob, y_pred = torch.max(y_mean, 1)
        y_confidence = []
        for y_p, y_s in zip(y_pred, y_std):
            y_confidence += [y_s[y_p]]
        return y_pred, y_prob, torch.stack(y_confidence)


