import torch
from tqdm import tqdm


class Ensemble:

    def __init__(self, model_class, trainer_factory, n_model, trainer_args={}, callbacks=[]):
        self.learners = []
        for i in range(n_model):
            t_args = dict(trainer_args)
            t_args['name'] = trainer_args['name'] + '_{}'.format(i) if 'name' in trainer_args else '{}'.format(i)
            self.learners.append(trainer_factory(model_class, **t_args))
        self.epochs_run = 0
        self.callbacks = callbacks

    def predict(self, x, device='cpu', return_prob=False, return_certainty=False, learner_args=None):
        """
        Predicting a data tensor.

        Args:
            x (torch.tensor): data
            device: device to run the model on
            return_prob: return probability or class label
            learner_args: additional learner arguments (this may not include the 'return_prob' argument)
            return_certainty: returns certainty about predictions

        Returns:
            prediction with certainty measure.
            return_prob = True:
                (Mean percentage, standard deviation of probability)
            return_prob = False:
                (Majority vote, percentage of learners voted for that label)

        """
        if learner_args is None:
            learner_args = {}

        preds = [leaner.forward(x, device=device) for leaner in self.learners]
        return torch.stack(preds, dim=1)

    def fit(self, epochs, device, checkpoint_int=10, validation_int=10, restore_early_stopping=False, verbose=1,
            callback_iter=-1):
        """
        Trains each learner of the ensemble for a number of epochs

        Args:
            callback_iter: number of iterations till the ensemble callbacks are called
            epochs: number of epochs to train each learner
            device: device to run the models on
            checkpoint_int: every checkpoint_int iterations the model is checkpointed
            validation_int:  every validation_int iterations the model is validated
            restore_early_stopping: restores the best performing weights after training
            verbose: verbosity

        Returns:
            None

        """
        # if callback_iter > 0:   # dive into a sequence of shorter training runs
        #     epoch_iter = [callback_iter for _ in range(epochs//callback_iter)]
        #     if epochs % callback_iter > 0:
        #         epoch_iter += [epochs % callback_iter]
        # else:
        #     epoch_iter = [epochs]
        #
        # for run_epochs in epoch_iter:
        #     for learner in self.learners:
        #         if verbose == 1:
        #             print('Trainer {}'.format(learner.name))
        #         learner.fit(epochs=run_epochs, device=device, checkpoint_int=checkpoint_int,
        #                     validation_int=validation_int, restore_early_stopping=restore_early_stopping)
        #     for cb in self.callbacks:
        #         cb.__call__(self)
        # for cb in self.callbacks:
        #     cb.__call__(self)
        for learner in self.learners:
            if verbose == 1:
                print('Trainer {}'.format(learner.name))
            learner.fit(epochs=epochs, device=device, checkpoint_int=checkpoint_int,
                        validation_int=validation_int, restore_early_stopping=restore_early_stopping)
        for cb in self.callbacks:
            cb.__call__(self)

    def dump_checkpoint(self, path=None, tag='checkpoint'):
        """
        Dumps a checkpoint for each learner.

        Args:
            path: source folder of the checkpoints
            tag: addition tags of the checkpoints

        Returns:
            None
        """
        for trainer in self.learners:
            trainer.dump_checkpoint(path=path, tag=tag)

    def load_checkpoint(self, path=None, tag='checkpoint', device='cpu'):
        """
        Loads a set of checkpoints, one for each learner

        Args:
            path: source folder of the checkpoints
            tag: addition tags of the checkpoints

        Returns:
            None

        """
        for learner in self.learners:
            learner.load_checkpoint(path=path, tag=tag, device=device)

    def run_validation(self, device='cpu'):
        """
        Validates the ensemble on the validation dataset.

        Args:
            device: device to run the evaluation on

        Returns:
            predicted and true labels for each learner

        """
        y_pred_learners = []
        y_true_learners = []

        for learner in self.learners:
            y_pred = []
            y_true = []
            for X, y in learner.val_loader:
                y_true += [y]
                y_pred += [learner.forward(X, device=device)]
            y_pred_learners += [torch.cat(y_pred)]
            y_true_learners += [torch.cat(y_true)]
        return y_pred_learners, y_true_learners

    def resume_training(self, epochs, device='cpu', checkpoint_int=10, validation_int=10, restore_early_stopping=False, verbose=1):
        """
        The primarily purpose of this method is to return to training after an interrupted trainings cycle of the ensemble.

        Args:
            epochs: max number of epochs to run the learners for
            device: device to run the models on
            checkpoint_int: every checkpoint_int iterations the model is checkpointed
            validation_int:  every validation_int iterations the model is validated
            restore_early_stopping: restores the best performing weights after training
            verbose: verbosity

        Returns:
            None

        """
        for learner in self.learners:
            train_epochs = epochs - learner.train_dict['epochs_run']
            if verbose == 1:
                print('Trainer {} - train for {} epochs'.format(learner.name, train_epochs))
            learner.fit(epochs=train_epochs, device=device, checkpoint_int=checkpoint_int,
                        validation_int=validation_int, restore_early_stopping=restore_early_stopping)

    def to(self, device):
        for learner in self.learners:
            learner.to(device)

    def eval(self):
        for learner in self.learners:
            learner.eval()


class BaysianEnsemble(Ensemble):

    def __init__(self, model_class, trainer_factory, n_model, trainer_args={}, callbacks=[]):
        super(BaysianEnsemble, self).__init__(model_class, trainer_factory, n_model, trainer_args=trainer_args, callbacks=callbacks)

    def predict(self, x, device='cpu'):
        self.to(device)
        self.eval()
        with torch.no_grad():
            return torch.stack([learner.forward(x, device=device) for learner in self.learners])

    @staticmethod
    def get_confidence(y_mean, y_std):
        y_prob, y_pred = torch.max(y_mean, 1)
        y_confidence = []
        for y_p, y_s in zip(y_pred, y_std):
            y_confidence += [y_s[y_p]]
        return y_pred, y_prob, torch.stack(y_confidence)


