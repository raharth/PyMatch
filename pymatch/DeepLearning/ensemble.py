import torch
from tqdm import tqdm
from pymatch.utils.exception import TerminationException


class Ensemble:

    def __init__(self, model_class, trainer_factory, n_model, trainer_args={}, callbacks=[], save_memory=False):
        self.learners = []
        for i in range(n_model):
            t_args = dict(trainer_args)
            t_args['name'] = trainer_args['name'] + '_{}'.format(i) if 'name' in trainer_args else '{}'.format(i)
            self.learners.append(trainer_factory(model_class, **t_args))
        self.epochs_run = 0
        self.callbacks = callbacks
        self.save_memory = save_memory
        self.train_dict = {'epochs_run': 0}
        self.dump_path = trainer_args['learner_args'].get('dump_path', 'tmp')
        self.training = True
        self.device = 'cpu'
        self.name = 'ensemble'

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
        return torch.stack(preds, dim=0)

    def fit(self, epochs, device, restore_early_stopping=False, verbose=1, learning_partition=0):
        """
        Trains each learner of the ensemble for a number of epochs

        Args:
            epochs:                     number of epochs to train each learner
        device:                         device to run the models on
            restore_early_stopping:     restores the best performing weights after training
            learning_partition:         how many steps to take at a time for a specific learner
            verbose:                    verbosity

        Returns:
            None

        """
        epoch_iter = self.partition_learning_steps(epochs, learning_partition)

        self.start_callbacks()

        for run_epochs in epoch_iter:
            for learner in self.learners:
                if verbose >= 1:
                    print('Trainer {}'.format(learner.name))
                try:
                    learner.fit(epochs=run_epochs,
                                device=device,
                                restore_early_stopping=restore_early_stopping)
                except TerminationException as te:
                    print(te)

                if self.save_memory and learning_partition < 1:     # this means there is no learning partition
                    del learner
                    torch.cuda.empty_cache()
            for cb in self.callbacks:
                try:
                    cb.__call__(self)
                except Exception as e:
                    print(f'Ensemble callback {cb} failed with exception:\n{e}')
                    raise e
            self.train_dict['epochs_run'] = self.train_dict.get('epochs_run', 0) + 1

    def start_callbacks(self):
        for cb in self.callbacks:
            cb.start(self)
        for learner in self.learners:
            for cb in learner.callbacks:
                cb.start(learner)

    def partition_learning_steps(self, epochs, learning_partition):
        """
        Divides the entire epochs to run into shorter trainings phases. This is used to train multiple agents
        simultaneously.
        Args:
            epochs:
            learning_partition:

        Returns:

        """
        if learning_partition > 0:
            epoch_iter = [learning_partition for _ in range(epochs // learning_partition)]
            if epochs % learning_partition > 0:
                epoch_iter += [epochs % learning_partition]
        else:
            epoch_iter = [epochs]
        return epoch_iter

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
        path = self.get_path(path=path, tag=tag)
        torch.save({'train_dict': self.train_dict}, path)

    def load_checkpoint(self, path=None, tag='checkpoint', device='cpu', secure=True):
        """
        Loads a set of checkpoints, one for each learner

        Args:
            path: source folder of the checkpoints
            tag: addition tags of the checkpoints

        Returns:
            None

        """
        for learner in self.learners:
            try:
                learner.load_checkpoint(path=path, tag=tag, device=device)
            except FileNotFoundError as e:
                if secure:
                    raise e
                else:
                    print(f'learner `{learner.name}` could not be found and is hence newly initialized')
        checkpoint = torch.load(self.get_path(path=path, tag=tag), map_location=device)
        self.train_dict = checkpoint.get('train_dict', self.train_dict)

    def fit_resume(self, epochs, **fit_args):
        self.load_checkpoint(path=f'{self.dump_path}/checkpoint')
        epochs = epochs - self.train_dict['epochs_run']
        return self.fit(epochs=epochs, **fit_args)

    # @todo depricated
    # def resume_training(self, epochs, device='cpu', restore_early_stopping=False, verbose=1):
    #     # self, epochs, device, restore_early_stopping=False, verbose=1, learning_partition=0
    #     """
    #     The primarily purpose of this method is to return to training after an interrupted trainings cycle of the ensemble.
    #
    #     Args:
    #         epochs: max number of epochs to run the learners for
    #         device: device to run the models on
    #         checkpoint_int: every checkpoint_int iterations the model is checkpointed
    #         validation_int:  every validation_int iterations the model is validated
    #         restore_early_stopping: restores the best performing weights after training
    #         verbose: verbosity
    #
    #     Returns:
    #         None
    #
    #     """
    #     for learner in self.learners:
    #         train_epochs = epochs - learner.train_dict['epochs_run']
    #         if train_epochs > 0:
    #             print('Trainer {} - train for {} epochs'.format(learner.name, train_epochs))
    #             learner.fit(epochs=train_epochs,
    #                         device=device,
    #                         restore_early_stopping=restore_early_stopping,
    #                         verbose=verbose)
    #         else:
    #             print(f'Trainer {learner.name} - was already trained')
    #
    #         if self.save_memory:
    #             del learner
    #             torch.cuda.empty_cache()

    def to(self, device):
        self.device = device
        for learner in self.learners:
            learner.to(device)

    def eval(self):
        self.training = False
        for learner in self.learners:
            learner.eval()

    def train(self):
        self.training = True
        for learner in self.learners:
            learner.train()

    def __call__(self, x, device='cpu'):
        return self.predict(x, device=device)

    def get_path(self, path, tag):
        """
        Returns the path for dumping or loading a checkpoint.

        Args:
            path: target folder
            tag: additional name tag

        Returns:

        """
        if path is None:
            path = self.dump_path
        return '{}/{}_{}.mdl'.format(path, tag, self.name)


# @ todo is this actually still in use?
class BaysianEnsemble(Ensemble):

    def __init__(self, model_class, trainer_factory, n_model, trainer_args={}, callbacks=[]):
        super(BaysianEnsemble, self).__init__(model_class, trainer_factory, n_model, trainer_args=trainer_args, callbacks=callbacks)

    def predict(self, x, device='cpu', eval=True):
        self.to(device)
        if eval:
            self.eval()
        else:
            self.train()
        with torch.no_grad():
            return torch.stack([learner.forward(x, device=device) for learner in self.learners])

    @staticmethod
    def get_confidence(y_mean, y_std):
        y_prob, y_pred = torch.max(y_mean, 1)
        y_confidence = []
        for y_p, y_s in zip(y_pred, y_std):
            y_confidence += [y_s[y_p]]
        return y_pred, y_prob, torch.stack(y_confidence)
