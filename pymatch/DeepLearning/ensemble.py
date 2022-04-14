import torch
from pymatch.utils.exception import TerminationException
from pymatch.DeepLearning.learner import Predictor


def _predictor_factory(Model, model_args, learner_args, name, *args, **kwargs):
    l_args = dict(learner_args)
    l_args['name'] = f"{learner_args['name']}_{name}"
    return Predictor(model=Model(**model_args), **l_args)


class EnsemblePredictor:
    def __init__(self, model_class, n_model, dump_path=None, trainer_factory=_predictor_factory, trainer_args=None, train=False):
        if dump_path is None:
            dump_path = 'tmp'
            print('Warning: Using default dump_path at `./tmp`')
        if trainer_args is None:
            trainer_args = {}
        self.learners = []
        for i in range(n_model):
            t_args = dict(trainer_args)
            t_args['name'] = trainer_args['name'] + '_{}'.format(i) if 'name' in trainer_args else '{}'.format(i)
            self.learners.append(trainer_factory(model_class, **t_args))

        self.dump_path = dump_path
        # self.dump_path = trainer_args.get('dump_path', 'tmp')
        self.device = 'cpu'
        self.name = 'ensemble'
        self.training = train

    def predict(self, x, device='cpu'):
        """
        Predicting a data tensor.

        Args:
            x (torch.tensor): data
            device: device to run the model on
            # learner_args: additional learner arguments (this may not include the 'return_prob' argument)

        Returns:
            prediction with certainty measure.
            return_prob = True:
                (Mean percentage, standard deviation of probability)
            return_prob = False:
                (Majority vote, percentage of learners voted for that label)

        """
        preds = [leaner.forward(x, device=device) for leaner in self.learners]
        return torch.stack(preds, dim=0)

    def load_checkpoint(self, path=None, tag='checkpoint', device='cpu', secure=True):
        """
        Loads a set of checkpoints, one for each learner

        Args:
            secure:     secure loading, throws exception if a learner fails to load, otherwise it will just print the
                        error. Missing agents will thereby just be skipped.
            device:     device to move the ensemble to
            path:       source folder of the checkpoints
            tag:        addition tags of the checkpoints

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

    def to(self, device):
        self.device = device
        for learner in self.learners:
            learner.to(device)

    def eval(self):
        self.training = False
        for learner in self.learners:
            learner.eval()

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
        return f'{path}/{tag}_{self.name}.mdl'


class Ensemble(EnsemblePredictor):

    def __init__(self, model_class, trainer_factory, n_model, trainer_args=None, callbacks=None, save_memory=False,
                 train=True, *args, **kwargs):
        if callbacks is None:
            callbacks = []
        if trainer_args is None:
            trainer_args = {}
        super().__init__(model_class=model_class,
                         trainer_factory=trainer_factory,
                         n_model=n_model,
                         trainer_args=trainer_args,
                         train=train,
                         *args,
                         **kwargs)
        # self.epochs_run = 0     # @todo deprecated remove this shit
        self.callbacks = callbacks
        self.save_memory = save_memory
        self.save_memory = save_memory
        self.train_dict = {'epochs_run': 0}

    def fit(self, epochs, device, verbose=1, learning_partition=0):
        """
        Trains each learner of the ensemble for a number of epochs

        Args:
            learning_partition:
            verbose:
            restore_early_stopping:
            device:
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
                                device=device)
                except TerminationException as te:
                    print(te)

                if self.save_memory and learning_partition < 1:     # this means there is no learning partition
                    del learner
                    torch.cuda.empty_cache()
            for cb in self.callbacks:
                try:
                    cb(self)
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
        for learner in self.learners:
            learner.dump_checkpoint(path=path, tag=tag)
        path = self.get_path(path=path, tag=tag)
        torch.save(self.create_state_dict(), path)

    def create_state_dict(self):
        state_dict = {'train_dict': self.train_dict}
        return state_dict

    def load_checkpoint(self, path=None, tag='checkpoint', device='cpu', secure=True):
        """
        Loads a set of checkpoints, one for each learner

        Args:
            path: source folder of the checkpoints
            tag: addition tags of the checkpoints

        Returns:
            None

        """
        super(Ensemble, self).load_checkpoint(path=path, tag='checkpoint', device='cpu', secure=True)
        self.restore_checkpoint(torch.load(self.get_path(path=path, tag=tag), map_location=device))

    def restore_checkpoint(self, checkpoint):
        self.train_dict = checkpoint.get('train_dict', self.train_dict)

    def fit_resume(self, epochs, **fit_args):
        self.load_checkpoint(path=f'{self.dump_path}/checkpoint')
        epochs = epochs - self.train_dict['epochs_run']
        return self.fit(epochs=epochs, **fit_args)

    def to(self, device):
        self.device = device
        for learner in self.learners:
            learner.to(device)

    def train(self):
        self.training = True
        for learner in self.learners:
            learner.train()


class DQNEnsemble(Ensemble):

    def __init__(self, memory, selection_strategy, env, player, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # todo this needs to be cleaned up
        self.memory = self.train_loader = memory
        self.selection_strategy = selection_strategy
        self.env = env
        self.player = player

    def play_episode(self):
        return self.player(self, self.selection_strategy, self.train_loader)

    def create_state_dict(self):
        state_dict = super().create_state_dict()
        state_dict['memory'] = self.train_loader.create_state_dict()
        return state_dict

    def restore_checkpoint(self, checkpoint):
        super().restore_checkpoint(checkpoint=checkpoint)
        self.memory = checkpoint.get('memory', self.memory)


class EfficientDQNEnsemble(DQNEnsemble):
    def __init__(self, max_uncertainty, init_samples, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_uncertainty = max_uncertainty
        self.init_samples = init_samples
        self.episode_count = 0

    def play_episode(self):
        if (len(self.memory) < self.init_samples) or (self.memory.memory['uncertainty'].mean() < self.max_uncertainty):
            self.train_dict['episode_count'] = self.train_dict.get('episode_count', 0) + 1
            self.train_dict['episode_sampled'] = self.train_dict.get('episode_sampled', []) + [self.train_dict['epochs_run']]
            print(f'Sampling episode {self.train_dict["episode_count"]}')
            super(EfficientDQNEnsemble, self).play_episode()
        else:
            print('Uncertainty was to high to sample new episodes')


class DuelingDQNEnsemble(DQNEnsemble):
    # @todo this is not just for dueling but all ensembles that output two tensors instead of one
    def predict(self, x, device='cpu'):
        """
        Predicting a data tensor.

        Args:
            x (torch.tensor): data
            device: device to run the model on
            # learner_args: additional learner arguments (this may not include the 'return_prob' argument)

        Returns:
            prediction with certainty measure.
            return_prob = True:
                (Mean percentage, standard deviation of probability)
            return_prob = False:
                (Majority vote, percentage of learners voted for that label)

        """
        preds = [leaner.forward(x, device=device) for leaner in self.learners]
        if isinstance(preds[0], tuple):
            return tuple(map(torch.stack, zip(*preds)))
        return torch.stack(preds, dim=0)