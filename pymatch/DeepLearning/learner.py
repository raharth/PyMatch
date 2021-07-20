import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod
import os
import shutil
import pandas as pd

from pymatch.utils import DataHandler


class Predictor(ABC):

    def __init__(self, model, name, dump_path='./tmp', device='cpu', train=False, **kwargs):
        """

        Args:
            model: model that can forward
            name: just a name
        """
        self.model = model
        self.name = name
        self.device = device
        self.dump_path = dump_path
        self.training = train
        if len(kwargs) > 0:
            print(f'There are unused and ignored kwargs: {kwargs.keys()}')

    def __call__(self, data, device=None):
        return self.forward(data=data, device=device)

    def forward(self, data, device=None, eval=True):
        """
        Predicting a batch as tensor.

        Args:
            data: data to forward
            device: device to run the model on

        Returns:
            prediction (, true label)
        """
        if device is None:
            device = self.device

        if eval:
            self.model.eval()
        else:
            self.model.train()
        self.model.to(device)
        data = data.to(device)
        y_pred = self.model.forward(data)
        return y_pred

    def predict(self, data, device='cpu'):
        """
        Predicting a batch as tensor.

        Args:
            data: data to forward
            device: device to run the model on

        Returns:
            prediction (, true label)
        """

        with torch.no_grad():
            self.model.eval()
            self.model.to(device)
            data = data.to(device)
            y_pred = self.model.forward(data)
            return y_pred

    def load_checkpoint(self, path, tag, device='cpu'):
        """
        Loads dumped checkpoint.

        Args:
            path: source path
            tag: additional name tag

        Returns:
            None

        """
        checkpoint = torch.load(self.get_path(path=path, tag=tag), map_location=device)
        self.restore_checkpoint(checkpoint)

    def restore_checkpoint(self, checkpoint):
        """
        Restores a checkpoint_dictionary.
        This should be redefined by every derived learner (if it introduces own members), while the derived learner should call the parent function

        Args:
            checkpoint: dictionary containing the state of the learner

        Returns:
            None
        """
        self.model.load_state_dict(state_dict=checkpoint['model_state_dict'])

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

    def eval(self):
        self.training = False
        self.model.eval()

    def train(self):
        self.training = True
        self.model.train()

    def to(self, device):
        self.device = device
        self.model.to(device)


class Learner(Predictor):

    def __init__(self, model, optimizer, crit, train_loader, grad_clip=None, name='', callbacks=None, dump_path='./tmp', device='cpu'):
        """

        Args:
            model:              neural network
            optimizer:          optimizer to optimize the model with
            crit:               loss gunction
            train_loader:       train loader
            grad_clip:          gradient clipping
            # load_checkpoint:    determines if checkpoint should be leaded
            name:               name of the learner, used for dumping
            callbacks:          list of callbacks
            dump_path:          path to dump the model to when saving. Many callbacks rely on it as well
            device:             device to run the learner on
        """
        super().__init__(model=model, name=name, device=device, dump_path=dump_path, train=True)
        self.optimizer = optimizer
        self.crit = crit

        self.grad_clip = grad_clip

        self.train_loader = train_loader
        self.callbacks = [] if callbacks is None else callbacks

        self.train_dict = {'train_losses': [],                  # list of all training losses
                           'epochs_run': 0,                     # number of epochs the model has been trained
                           'best_train_performance': np.inf,    # best training performance
                           'best_val_performance': np.inf,      # best training performance
                           'epochs_since_last_val_improvement': 0   # @todo ugly shit, shouldnt be here
                           }

    def _backward(self, loss):
        """
        Backward pass for the model, also performing a grad clip if defined for the learner.

        Args:
            loss: loss the backward pass is based on

        Returns:
            None

        """
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

    def dump_checkpoint(self, path=None, tag='checkpoint'):
        """
        Dumping a checkpoint of the model.

        Args:
            path: target folder path
            tag: addition tag for the dump

        Returns:
            None

        """
        path = self.get_path(path=path, tag=tag)
        torch.save(self.create_state_dict(), path)

    def create_state_dict(self):
        """
        Creates the state dictionary of a learner.
        This should be redefined by each derived learner that introduces own members. Always call the parents method.
        This dictionary can then be extended by
        the derived learner's members

        Returns:
            state dictionary of the learner

        """
        state_dict = {'model_state_dict': self.model.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict(),
                      'train_dict': self.train_dict,
                      }
        return state_dict

    def restore_checkpoint(self, checkpoint):
        """
        Restores a checkpoint_dictionary.
        This should be redefined by every derived learner (if it introduces own members), while the derived learner should call the parent function

        Args:
            checkpoint: dictionary containing the state of the learner

        Returns:
            None
        """
        super(Learner, self).restore_checkpoint(checkpoint=checkpoint)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def fit(self, epochs, device, verbose=1):
        """
        Trains the learner for a number of epochs.

        Args:
            epochs: number of epochs to train
            device: device to runt he model on
            verbose: verbosity

        Returns:
            None

        """
        self.device = device
        self.init_callbacks()

        for epoch in range(epochs):

            self.train_dict['epochs_since_last_val_improvement'] += 1   # @todo this shouldn't be here... since is is related to the callback

            if verbose == 1:
                name = '' if self.name == '' else ' - name: {}'.format(self.name)
                print('\nepoch: {}{}'.format(self.train_dict['epochs_run'], name), flush=True)

            train_loss = self.fit_epoch(device)

            # tracking training performance
            if train_loss < self.train_dict.get('best_train_performance', -np.inf):
                self.train_dict['best_train_performance'] = train_loss

            self.train_dict['epochs_run'] += 1

            for cb in self.callbacks:
                try:
                    cb(model=self)
                except Exception as e:
                    print(f'callback {cb} failed with exception\n{e}')

    @abstractmethod
    def fit_epoch(self, device, verbose=1):
        """
        Train a single epoch.
        this has to be implemented by each type of learner individually.

        Args:
            device: device to run it on 'cpu' or 'cuda'
            verbose: verbosity of the learning

        Returns:
            current loss
        """
        raise NotImplementedError

    def init_callbacks(self):
        for cb in self.callbacks:
            cb.start(self)


class ClassificationLearner(Learner):

    def __init__(self,
                 model,
                 optimizer,
                 crit,
                 train_loader,
                 grad_clip=None,
                 name='',
                 callbacks=None,
                 dump_path='./tmp',
                 **kwargs):
        super(ClassificationLearner, self).__init__(model=model,
                                                    optimizer=optimizer,
                                                    crit=crit,
                                                    train_loader=train_loader,
                                                    grad_clip=grad_clip,
                                                    name=name,
                                                    callbacks=callbacks,
                                                    dump_path=dump_path,
                                                    **kwargs)
        self.train_dict['train_accuracy'] = []
        self.train_dict['val_accuracy'] = []

    def fit_epoch(self, device, verbose=1):
        """
        Train a single epoch.

        Args:
            device: device t-o run it on 'cpu' or 'cuda'
            verbose: verbosity of the learning

        Returns:
            current loss
        """
        self.model.train()
        self.model.to(device)

        accuracies = []
        losses = []

        for batch, (data, labels) in tqdm(enumerate(self.train_loader)):
            data = data.to(device)
            labels = labels.to(device)

            y_pred = self.model.forward(data)
            loss = self.crit(y_pred, labels)

            self._backward(loss)
            losses += [loss.item()]
            accuracies += [(y_pred.max(dim=1)[1] == labels)]
        loss = np.mean(losses)
        self.train_dict['train_losses'] += [loss]
        accuracy = torch.cat(accuracies).float().mean().item()
        self.train_dict['train_accuracy'] += [accuracy]
        if verbose == 1:
            print('train loss: {:.4f} - train accuracy: {:.4f}'.format(loss, accuracy))
        return loss


class RegressionLearner(Learner):

    def __init__(self,
                 model,
                 optimizer,
                 crit,
                 train_loader,
                 grad_clip=None,
                 name='',
                 callbacks=None,
                 dump_path='./tmp',
                 **kwargs):
        super(RegressionLearner, self).__init__(model=model,
                                                optimizer=optimizer,
                                                crit=crit,
                                                train_loader=train_loader,
                                                grad_clip=grad_clip,
                                                name=name,
                                                callbacks=callbacks,
                                                dump_path=dump_path,
                                                **kwargs)

    def fit_epoch(self, device, verbose=1):
        """
        Train a single epoch.

        Args:
            device: device t-o run it on 'cpu' or 'cuda'
            verbose: verbosity of the learning

        Returns:
            current loss
        """
        self.model.fit()
        self.model.to(device)

        losses = []

        for batch, (data, labels) in tqdm(enumerate(self.train_loader)):
            data = data.to(device)

            y_pred = self.model.forward(data).to('cpu')
            loss = self.crit(y_pred, labels)

            self._backward(loss)
            losses += [loss.item()]
        loss = np.mean(losses)
        self.train_dict['train_losses'] += [loss]
        if verbose == 1:
            print('train loss: {:.4f}'.format(loss))
        return loss
