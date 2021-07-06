import torch
from pymatch.utils.exception import TerminationException
from pymatch.DeepLearning.learner import Predictor


class EnsemblePredictor:
    def __init__(self, model_class, n_model, trainer_factory=Predictor, trainer_args=None, train=False):
        """
        Base class for the Ensemble. If one wants  to train an ensemble one has to use the regular Ensemble class.
        This class is only meant to be used with an already trained ensemble and only serves the purpose of being able
        to apply the ensemble to data.

        Args:
            model_class:        class that defines the model architecture and forward pass. Has to be derived from
                                torch.nn.Module
            n_model:            number of individual learners that are part of the ensemble
            trainer_factory:    factory function/object, that generates a full learner. If this is only used as a
                                predictor one does not need to provide any factory function. This is only necessary
                                if this class is used as the super class of an other ensemble
            trainer_args:       arguments, used to call the factory with. In the case of using this only as predictor
                                this can be left empty.
            train:              bool that defines if the model is set to train or eval mode
        """
        if trainer_args is None:
            trainer_args = {}
        self.learners = []
        for i in range(n_model):
            t_args = dict(trainer_args)
            t_args['name'] = trainer_args['name'] + '_{}'.format(i) if 'name' in trainer_args else '{}'.format(i)
            self.learners.append(trainer_factory(model_class, **t_args))

        self.dump_path = trainer_args['learner_args'].get('dump_path', 'tmp')
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
        """
        Moves the ensemble and all learners to to given device.

        Args:
            device: device to move the ensemble to

        Returns:

        """
        self.device = device
        for learner in self.learners:
            learner.to(device)

    def eval(self):
        """
        Sets the ensemble and all learners that are part of it to eval mode.

        Returns:

        """
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
            The path to store or load the model to/from
        """
        if path is None:
            path = self.dump_path
        return f'{path}/{tag}_{self.name}.mdl'


class Ensemble(EnsemblePredictor):

    def __init__(self, model_class, trainer_factory, n_model, trainer_args=None, callbacks=None, save_memory=False,
                 train=True):
        """
        Ensemble around a number of learners, that can be trained and called as a individual agent.

        Args:
            model_class:        class that defines the model architecture and forward pass. Has to be derived from
                                torch.nn.Module
            trainer_factory:    factory function/object, that generates a full learner, containing a model, loss, optim
            n_model:            number of individual learners that are part of the ensemble
            trainer_args:       arguments, used to call the factory with
            callbacks:          list of callbacks that can be called from the ensemble
            save_memory:        if set to true the ensemble deletes each agent after it was trained. This is useful to
                                free up VRAM during training
            train:              bool that defines if the model is set to train or eval mode
        """
        if callbacks is None:
            callbacks = []
        if trainer_args is None:
            trainer_args = {}
        super().__init__(model_class=model_class,
                         trainer_factory=trainer_factory,
                         n_model=n_model,
                         trainer_args=trainer_args,
                         train=train)
        self.epochs_run = 0
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
                    cb.__call__(self)
                except Exception as e:
                    print(f'Ensemble callback {cb} failed with exception:\n{e}')
                    raise e
            self.train_dict['epochs_run'] = self.train_dict.get('epochs_run', 0) + 1

    def start_callbacks(self):
        """
        Calling this function starts all callbacks part of the ensemble itself and all learners below it.

        Returns:

        """
        for cb in self.callbacks:
            cb.start(self)
        for learner in self.learners:
            for cb in learner.callbacks:
                cb.start(learner)

    def partition_learning_steps(self, epochs, learning_partition):
        """
        Divides the entire epochs to run into shorter trainings phases. This is used to train multiple agents
        "simultaneously". This is useful if you want to evaluate the ensemble during training as a hole.

        Args:
            epochs:                 number of epochs that have to be run
            learning_partition:     number of epochs before the next agent gets trained

        Returns:
            List of number of epochs to be run at a time, summing up to the full number of required epochs
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
        Dumps a checkpoint for each learner and the ensemble itself.

        Args:
            path: source folder of the checkpoints
            tag: addition tags of the checkpoints

        Returns:
            None
        """
        for trainer in self.learners:
            trainer.dump_checkpoint(path=path, tag=tag)
        path = self.get_path(path=path, tag=tag)
        torch.save(self.create_state_dict(), path)

    def create_state_dict(self):
        """
        Creates a state dict for the entire ensemble.

        Returns:

        """
        state_dict = {'train_dict': self.train_dict}
        return state_dict

    def load_checkpoint(self, path=None, tag='checkpoint', device='cpu', secure=True):
        """
        Loads a set of checkpoints, one for each learner

        Args:
            path:   source folder of the checkpoints
            tag:    addition tags of the checkpoints

        Returns:
            None

        """
        super(Ensemble, self).load_checkpoint(path=None, tag='checkpoint', device='cpu', secure=True)
        self.restore_checkpoint(torch.load(self.get_path(path=path, tag=tag), map_location=device))

    def restore_checkpoint(self, checkpoint):
        """
        Restores a checkpoint from the given path-

        Args:
            checkpoint: path to the checkpoint

        Returns:

        """
        self.train_dict = checkpoint.get('train_dict', self.train_dict)

    def fit_resume(self, epochs, **fit_args):
        """
        Resumes training after it was interrupted, or can continue with learning after a previous `fit` call. This is
        especially useful if the training was interrupted and you want all agents to be trained a specific number of
        epochs.

        Args:
            epochs:         full number of epochs it has to be trained
            **fit_args:     arguments used to call the `fit` method

        Returns:

        """
        self.load_checkpoint(path=f'{self.dump_path}/checkpoint')
        epochs = epochs - self.train_dict['epochs_run']
        return self.fit(epochs=epochs, **fit_args)

    def train(self):
        """
        Sets the entire ensemble including its individual agents to training mode

        Returns:

        """
        self.training = True
        for learner in self.learners:
            learner.train()


class DQNEnsemble(Ensemble):

    def __init__(self, memory, selection_strategy, env, player, *args, **kwargs):
        """
        Ensemble of DQN agents. This assumes to have a single unique memory, potentially filled and use by the agents
        together. If the ensemble is not meant to train as a single instance on a common memory that is generated by the
        ensembles instead of the individual agents, one can also use the regular Ensemble.

        Args:
            memory:                 memory of the ensemble
            selection_strategy:     action selection strategy used by the ensemble when playing
            env:                    environment to interact with
            player:                 Player that defines how an episodes is played by the ensemble as well as which and
                                    how memories are stored.
            *args:                  Arguments to pass to super class
            **kwargs:               Arguments to pass to super class
        """
        super().__init__(*args, **kwargs)
        # todo this needs to be cleaned up
        self.train_loader = memory
        self.memory = memory
        self.selection_strategy = selection_strategy
        self.env = env
        self.player = player

    def play_episode(self):
        """
        Plays an episode using the player provided to the ensemble

        Returns:

        """
        return self.player(self, self.selection_strategy, self.train_loader)

    def create_state_dict(self):
        """
        Creates a state dictionary, also storing the ensembles memory
        Returns:

        """
        state_dict = super().create_state_dict()
        state_dict['memory'] = self.train_loader.create_state_dict()
        return state_dict

    def restore_checkpoint(self, checkpoint):
        """
        Restores a checkpoint including the memory from the provided path.

        Args:
            checkpoint: path to the checkpoint file.

        Returns:

        """
        super().restore_checkpoint(checkpoint=checkpoint)
        self.memory = checkpoint.get('memory', self.memory)


