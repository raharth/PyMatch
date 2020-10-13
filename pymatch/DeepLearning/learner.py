import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod
import os
import shutil
import pandas as pd

from pymatch.utils import DataHandler


class Learner(ABC):

    def __init__(self,
                 model,
                 optimizer,
                 crit,
                 train_loader,
                 grad_clip=None,
                 load_checkpoint=False,
                 name='',
                 callbacks=None,
                 dump_path='./tmp',
                 device='cpu'
                 ):
        """

        Args:
            model:              neural network
            optimizer:          optimizer to optimize the model with
            crit:               loss gunction
            train_loader:       train loader
            grad_clip:          gradient clipping
            load_checkpoint:    determines if checkpoint should be leaded
            name:               name of the learner, used for dumping
            callbacks:          list of callbacks
            dump_path:          path to dump the model to when saving. Many callbacks rely on it as well
            device:             device to run the learner on
        """
        self.training = True
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.crit = crit

        self.grad_clip = grad_clip

        self.train_loader = train_loader
        self.dump_path = dump_path
        self.checkpoint_path = f'{dump_path}/checkpoint'
        self.early_stopping_path = f'{dump_path}/early_stopping'

        # creating folders
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        if not os.path.exists(self.early_stopping_path):
            os.makedirs(self.early_stopping_path)

        self.name = name  # name for the learner used for checkpointing and early stopping
        self.callbacks = [] if callbacks is None else callbacks

        self.train_dict = {'train_losses': [],                  # list of all training losses
                           # 'val_losses': [],                    # list of all validation losses
                           # 'val_epochs': [],                    # list of validated epochs
                           'epochs_run': 0,                     # number of epochs the model has been trained
                           'best_train_performance': np.inf,    # best training performance
                           'best_val_performance': np.inf,    # best training performance
                           'epochs_since_last_val_improvement': 0   # @todo ugly shit, shouldnt be here
                           }

        if load_checkpoint:
            self.load_checkpoint(self.checkpoint_path, tag='checkpoint')

    def __call__(self, data, device=None):
        return self.forward(data=data, device=device)

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
        # loss.clone().backward(retain_graph=True)
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
                      # 'loss': self.losses,
                      # 'val_loss': self.val_losses,
                      # 'val_epoch': self.val_epochs,
                      # 'best_train_performance': self.best_train_performance,
                      # 'epochs_since_last_val_improvement': self.epochs_since_last_train_improvement
                      }
        return state_dict

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
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_dict = checkpoint['train_dict']

    def get_path(self, path, tag):
        """
        Returns the path for dumping or loading a checkpoint.

        Args:
            path: target folder
            tag: additional name tag

        Returns:

        """
        if path is None:
            path = self.checkpoint_path
        return '{}/{}_{}.mdl'.format(path, tag, self.name)

    def fit(self, epochs, device, restore_early_stopping=False, verbose=1):
        """
        Trains the learner for a number of epochs.

        Args:
            epochs: number of epochs to train
            device: device to runt he model on
            restore_early_stopping: restores best performing weights after training
            verbose: verbosity

        Returns:
            None

        """
        self.device = device

        for epoch in range(epochs):

            self.train_dict['epochs_since_last_val_improvement'] += 1   # @todo this shouldn't be here... since is is related to the callback

            if verbose == 1:
                name = '' if self.name == '' else ' - name: {}'.format(self.name)
                print('\nepoch: {}{}'.format(self.train_dict['epochs_run'], name))

            train_loss = self.fit_epoch(device)

            # tracking training performance
            if train_loss < self.train_dict['best_train_performance']:
                self.train_dict['best_train_performance'] = train_loss

            self.train_dict['epochs_run'] += 1

            for cb in self.callbacks:
                try:
                    cb(model=self)
                except Exception as e:
                    print(f'callback {cb} failed with exception\n{e}')

        if restore_early_stopping:
            self.load_checkpoint(self.early_stopping_path, 'early_stopping')
        self.dump_checkpoint(self.checkpoint_path)

    def eval(self):
        self.training = False
        self.model.eval()

    def train(self):
        self.training = True
        self.model.train()

    def to(self, device):
        self.model.to(device)

    def forward(self, data, device=None, eval=True):
        if device is None:
            device = self.device
    # def predict(self, data, device='cpu'):
        """
        Predicting a batch as tensor.

        Args:
            data: data to forward
            device: device to run the model on

        Returns:
            prediction (, true label)
        """
        with torch.no_grad():
            if eval:
                self.model.eval()
            else:
                self.model.train()
            self.model.to(device)
            data = data.to(device)
            y_pred = self.model.forward(data)
            return y_pred

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

    # @abstractmethod
    # def validate(self, device, verbose=0):
    #     """
    #     Validation of the validation data if provided
    #
    #     Args:
    #         device: device to run it on 'cpu' or 'cuda'
    #         verbose: verbosity of the learning
    #
    #     Returns:
    #         loss
    #     """
    #     raise NotImplementedError


class ClassificationLearner(Learner):

    def __init__(self,
                 model,
                 optimizer,
                 crit,
                 train_loader,
                 grad_clip=None,
                 load_checkpoint=False,
                 name='',
                 callbacks=None,
                 dump_path='./tmp'):
        super(ClassificationLearner, self).__init__(model,
                                                    optimizer,
                                                    crit,
                                                    train_loader,
                                                    grad_clip,
                                                    load_checkpoint,
                                                    name,
                                                    callbacks=callbacks,
                                                    dump_path=dump_path)
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

    # def validate(self, device, verbose=0):
    #     """
    #     Validate the model on the validation data.
    #
    #     Args:
    #         device: device to run the model on
    #         verbose: verbosity
    #
    #     Returns:
    #         validation loss
    #
    #     """
    #     with torch.no_grad():
    #         self.eval()
    #         self.model.to(device)
    #         loss = []
    #         accuracies = []
    #         for data, y in self.val_loader:
    #             data = data.to(device)
    #             y = y
    #             y_pred = self.model(data)
    #             loss += [self.crit(y_pred.to('cpu'), y)]
    #
    #             y_pred = y_pred.max(dim=1)[1].to('cpu')
    #             accuracies += [(y_pred == y).float()]
    #
    #         loss = torch.stack(loss).mean().item()
    #         accuracy = torch.cat(accuracies).mean().item()
    #         self.train_dict['val_accuracy'] += [accuracy]
    #         if verbose == 1:
    #             print('val loss: {:.4f} - val accuracy: {:.4f}'.format(loss, accuracy))
    #         return loss


class RegressionLearner(Learner):

    def __init__(self,
                 model,
                 optimizer,
                 crit,
                 train_loader,
                 val_loader=None,
                 grad_clip=None,
                 load_checkpoint=False,
                 name='',
                 callbacks=None,
                 dump_path='./tmp'):
        super(RegressionLearner, self).__init__(model=model,
                                                optimizer=optimizer,
                                                crit=crit,
                                                train_loader=train_loader,
                                                grad_clip=grad_clip,
                                                load_checkpoint=load_checkpoint,
                                                name=name,
                                                callbacks=callbacks,
                                                dump_path=dump_path)

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

    def validate(self, device, verbose=0):
        """
        Validate the model on the validation data.

        Args:
            device: device to run the model on
            verbose: verbosity

        Returns:
            validation loss

        """
        with torch.no_grad():
            self.eval()
            self.model.to(device)
            loss = []
            for data, y in self.val_loader:
                data = data.to(device)
                y_pred = self.model(data).to('cpu')
                loss += [self.crit(y_pred, y)]

            loss = torch.stack(loss).mean().item()
            if verbose == 1:
                print('val loss: {:.4f}'.format(loss))
            return loss


# class ImageClassifier(ClassificationLearner):
#
#     def __init__(self,
#                  model,
#                  optimizer,
#                  crit,
#                  train_loader,
#                  grad_clip=None,
#                  load_checkpoint=False,
#                  name='',
#                  callbacks=None,
#                  dump_path='./tmp'
#                  ):
#         super(ImageClassifier, self).__init__(model,
#                                               optimizer,
#                                               crit,
#                                               train_loader,
#                                               grad_clip=grad_clip,
#                                               load_checkpoint=load_checkpoint,
#                                               name=name,
#                                               callbacks=callbacks,
#                                               dump_path=dump_path)
#
#     def create_result_df(self, data_loader, device='cpu'):
#         # y_pred, y_true = self.predict_data_loader(data_loader, device=device, return_true=True)
#         y_pred = DataHandler.predict_data_loader(self.model, data_loader, device=device)
#         data = np.stack((np.array(data_loader.dataset.samples)[:, 0], y_pred), 1)
#         return pd.DataFrame(data, columns=['img_path', 'label'])
#
#     @staticmethod
#     def sort_images(result_df, classes, output_root, class_mapping):
#         """
#         Sorting the images of the result DataFrame according to their predicted label.
#
#         !!! This may not be a torch.Subset!!!
#
#         Args:
#             result_df: result DataFrame that contains the path to the image and the assigned label
#             classes: all possible classes
#             output_root: root path for the output folders
#             class_mapping: mapping from output node to class label
#
#         Returns:
#
#         """
#         # creating output folders
#         for class_name in classes:
#             class_path = '{}/{}'.format(output_root, class_name)
#             if not os.path.exists(class_path):
#                 os.makedirs(class_path)
#
#         for img_path, label in result_df.values:
#             shutil.copy(img_path, '{}/{}'.format(output_root, class_mapping[label]))
