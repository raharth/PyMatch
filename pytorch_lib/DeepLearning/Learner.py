import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod
import os
import shutil
import pandas as pd


class Learner(ABC):

    def __init__(self, model, optimizer, crit, train_loader, val_loader=None, grad_clip=None, load_checkpoint=False, name='', callbacks=None):
        self.model = model  # neural network
        self.optimizer = optimizer  # optimizer for the network
        self.crit = crit  # loss

        self.grad_clip = grad_clip

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.checkpoint_path = './tmp/checkpoint'  # .pth'
        self.early_stopping_path = './tmp/early_stopping'  # .pth'

        # creating folders
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        if not os.path.exists(self.early_stopping_path):
            os.makedirs(self.early_stopping_path)

        self.name = name  # name for the learner used for checkpointing and early stopping
        self.callbacks = [] if callbacks is None else callbacks

        self.train_dict = {'train_losses': [],  # list of all training losses
                           'val_losses': [],    # list of all validation losses
                           'val_epochs': [],    # list of validated epochs
                           'epochs_run': 0,     # number of epochs the model has been trained
                           'best_val_performance': np.inf,  # best validation performance
                           'best_train_performance': np.inf,    # best training performance
                           'epochs_since_last_train_improvement': 0,
                           }

        # self.losses = []  # list of all training losses
        # self.val_losses = []  # list of all validation losses
        # self.val_epochs = []  # list of validated epochs
        # self.epochs_run = 0  # numer of epochs the model has been trained
        # self.best_val_performance = np.inf  # best validation performance
        # self.best_train_performance = np.inf  # best training performance
        # self.epochs_since_last_train_improvement = 0

        if load_checkpoint:
            self.load_checkpoint(self.checkpoint_path, tag='checkpoint')

    def backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

    def dump_checkpoint(self, path=None, tag='checkpoint'):
        path = self.get_path(path=path, tag=tag)
        torch.save(self.create_state_dict(), path)

    def create_state_dict(self):
        """
        Creates the state dictionary of a learner.
        This should be redefined by each derived learner that introduces own members. Always call the parents method. This dictionary can then be extended by
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
                      # 'epochs_since_last_train_improvement': self.epochs_since_last_train_improvement
                      }
        return state_dict

    def load_checkpoint(self, path, tag):
        checkpoint = torch.load(self.get_path(path=path, tag=tag))
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
        # self.epochs_run = checkpoint['epoch']
        # self.losses = checkpoint['loss']
        # self.val_losses = checkpoint['val_loss']
        # self.val_epochs = checkpoint['val_epoch']
        # self.best_train_performance = checkpoint['best_train_performance']
        # self.epochs_since_last_train_improvement = checkpoint['epochs_since_last_train_improvement']

    def get_path(self, path, tag):
        if path is None:
            path = self.checkpoint_path
        return '{}/{}_{}'.format(path, tag, self.name)

    def train(self, epochs, device, checkpoint_int=10, validation_int=10, restore_early_stopping=False, early_termination=-1, verbose=1):
        for epoch in range(epochs):
            self.train_dict['epochs_run'] += 1
            self.train_dict['epochs_since_last_train_improvement'] += 1

            if verbose == 1:
                name = '' if self.name == '' else ' - name: {}'.format(self.name)
                print('\nepoch: {}{}'.format(self.train_dict['epochs_run'], name))

            train_loss = self.train_epoch(device)

            # tracking training performance
            if train_loss < self.train_dict['best_train_performance']:
                self.train_dict['best_train_performance'] = train_loss
                self.train_dict['epochs_since_last_train_improvement'] = 0

            # checkpointing
            if epoch % checkpoint_int == 0:
                self.dump_checkpoint()

            # tracking validation performance
            if epoch % validation_int == 0 and self.val_loader is not None and validation_int > 0:
                if verbose == 1:
                    print('evaluating')
                val_loss = self.validate(device=device, verbose=verbose)
                self.train_dict['val_losses'] += [val_loss]
                self.train_dict['val_epochs'] += [self.train_dict['epochs_run']]
                if val_loss < self.train_dict['best_val_performance']:
                    self.train_dict['best_val_performance'] = val_loss
                    self.dump_checkpoint(path=self.early_stopping_path, tag='early_stopping')

            # early termination
            if 0 < early_termination < self.train_dict['epochs_since_last_train_improvement']:
                break

            for cb in self.callbacks:
                cb.callback(self)

        if restore_early_stopping:
            self.load_checkpoint(self.early_stopping_path, 'early_stopping')
        self.dump_checkpoint(self.checkpoint_path)

    def predict_data_loader(self, data_loader, device='cpu', return_true=False, model_args={}):
        y_pred = []
        y_true = []
        for X, y in data_loader:
            y_true += [y]
            y_pred += [self.predict(X, device=device, **model_args)]
        y_pred = torch.cat(y_pred)
        y_true = torch.cat(y_true)
        if return_true:
            return y_pred, y_true
        return y_pred

    @abstractmethod
    def train_epoch(self, device, verbose=1):
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

    @abstractmethod
    def predict(self, data, device, prob=False):
        """
        Predicting a batch of data.

        Args:
            data: batch of data to predict
            device: device to run it on 'cpu' or 'cuda'
            prob: @todo this is not supposed to be in the abstract class, since only useful for a classification learner

        Returns:
            predicted values for the given data
        """
        raise NotImplementedError

    @abstractmethod
    def validate(self, device, verbose=0):
        """
        Validation of the validation data if provided

        Args:
            device: device to run it on 'cpu' or 'cuda'
            verbose: verbosity of the learning

        Returns:
            loss
        """
        raise NotImplementedError


class ClassificationLearner(Learner):

    def __init__(self, model, optimizer, crit, train_loader, val_loader=None, grad_clip=None, load_checkpoint=False, name='', callbacks=None):
        super(ClassificationLearner, self).__init__(model, optimizer, crit, train_loader, val_loader, grad_clip,
                                                    load_checkpoint, name, callbacks=callbacks)
        self.train_dict['train_accuracy'] = []
        self.train_dict['val_accuracy'] = []

    def train_epoch(self, device, verbose=1):
        self.model.train()
        accuracies = []
        losses = []
        for batch, (data, labels) in tqdm(enumerate(self.train_loader)):
            data = data.to(device)
            labels = labels.to(device)
            y_pred = self.model.forward(data)
            loss = self.crit(y_pred, labels)
            self.backward(loss)
            if verbose == 1:    # somehow ugly, tis is only necessary if verbosity==1, but it is not outputting anything right here
                losses += [loss.item()]
                accuracies += [(y_pred.max(dim=1)[1] == labels)]
        loss = np.mean(losses)
        self.train_dict['losses'] += [loss]
        accuracy = torch.cat(accuracies).float().mean().item()
        self.train_dict['train_accuracy'] += [accuracy]
        if verbose == 1:
            print('train loss: {:.4f} - train accuracy: {}'.format(loss, accuracy))
        return loss

    def predict(self, data, device='cpu', return_prob=False):
        with torch.no_grad():
            self.model.eval()
            data = data.to(device)
            y_pred = self.model.forward(data).to('cpu')
            if return_prob:
                y_pred = y_pred # .data  # @todo 'data' necessary?
            else:
                y_pred = y_pred.max(dim=1)[1]
            return y_pred

    def validate(self, device, verbose=0):
        with torch.no_grad():
            self.model.eval()
            loss = []
            accuracies = []
            for data, y in self.val_loader:
                data = data.to(device)
                y_pred = self.model(data).to('cpu')
                loss += [self.crit(y_pred, y)]

                y_pred = y_pred.max(dim=1)[1]
                accuracies += [(y_pred == y).float()]

            loss = torch.stack(loss).mean().item()
            accuracy = torch.cat(accuracies).mean().item()
            self.train_dict['val_accuracy'] += [accuracy]
            if verbose == 1:
                print('val loss: {:.4f} - val accuracy: {:.4f}'.format(loss, accuracy))
            return loss

    # def create_state_dict(self):
    #     state_dict = super().create_state_dict()
    #     return state_dict
    #
    # def restore_checkpoint(self, checkpoint):
    #     super().restore_checkpoint(checkpoint)


class ImageClassifier(ClassificationLearner):

    def __init__(self, model, optimizer, crit, train_loader, val_loader=None, grad_clip=None, load_checkpoint=False, name='', callbacks=None):
        super(ImageClassifier, self).__init__(model, optimizer, crit, train_loader, val_loader=val_loader, grad_clip=grad_clip, load_checkpoint=load_checkpoint,
                                              name='', callbacks=callbacks)

    def create_result_df(self, data_loader, device='cpu'):
        y_pred, y_true = self.predict_data_loader(data_loader, device=device, return_true=True)
        data = np.stack((np.array(data_loader.dataset.samples)[:, 0], y_pred), 1)
        return pd.DataFrame(data, columns=['img_path', 'label'])

    @staticmethod
    def sort_images(create_result_df, classes, output_root, class_mapping):
        """
        Sorting the images of the result DataFrame according to their predicted label.

        !!! This may not be a torch.Subset!!!

        Args:
            create_result_df: result DataFrame that contains the path to the image and the assigned label
            classes: all possible classes
            output_root: root path for the output folders
            class_mapping: mapping from output node to class label

        Returns:

        """
        # creating output folders
        for class_name in classes:
            class_path = '{}/{}'.format(output_root, class_name)
            if not os.path.exists(class_path):
                os.makedirs(class_path)

        for img_path, label in create_result_df.values:
            shutil.copy(img_path, '{}/{}'.format(output_root, class_mapping[label]))
