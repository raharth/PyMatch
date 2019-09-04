import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod
import os


class Learner(ABC):

    def __init__(self, model, optimizer, crit, train_loader, val_loader=None, grad_clip=None, load_checkpoint=False,
                 name=''):
        self.model = model
        self.optimizer = optimizer
        self.crit = crit

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

        self.name = name

        self.losses = []
        self.val_losses = []
        self.val_epochs = []
        self.epochs_run = 0
        self.best_performance = np.inf

        if load_checkpoint:
            self.load_checkpoint(self.checkpoint_path, tag='checkpoint')

    def backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

    def dump_checkpoint(self, epoch, path=None, tag='checkpoint'):
        path = self.get_path(path=path, tag=tag)
        torch.save({'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': self.losses,
                    },
                   path)

    def load_checkpoint(self, path, tag):
        checkpoint = torch.load(self.get_path(path=path, tag=tag))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epochs_run = checkpoint['epoch']
        self.losses = checkpoint['loss']

    def get_path(self, path, tag):
        if path is None:
            path = self.checkpoint_path
        return '{}/{}_{}'.format(path, tag, self.name)

    def train(self, epochs, device, checkpoint_int=10, validation_int=10, restore_early_stopping=False, verbose=1):
        for epoch in range(epochs):
            self.epochs_run += 1
            if verbose == 1:
                name = '' if self.name == '' else ' - name: {}'.format(self.name)
                print('epoch: {}{}'.format(self.epochs_run, name))

            self.train_epoch(device)

            if epoch % checkpoint_int == 0:
                self.dump_checkpoint(self.epochs_run + epoch)

            if epoch % validation_int == 0 and self.val_loader is not None and validation_int > 0:
                if verbose == 1:
                    print('evaluating')
                performance = self.validate(device=device)
                self.val_losses += [performance]
                self.val_epochs += [self.epochs_run]
                if performance < self.best_performance:
                    self.best_performance = performance
                    self.dump_checkpoint(epoch=self.epochs_run + epoch, path=self.early_stopping_path,
                                         tag='early_stopping')
                if verbose == 1:
                    print('validation loss: {}\n'.format(performance))

        if restore_early_stopping:
            self.load_checkpoint(self.early_stopping_path, 'early_stopping')
        self.dump_checkpoint(self.epochs_run, self.checkpoint_path)

    @abstractmethod
    def train_epoch(self, device, verbose=1):
        raise NotImplementedError

    @abstractmethod
    def predict(self, data_loader, device, prob=False):
        raise NotImplementedError

    @abstractmethod
    def validate(self, device, verbose=0):
        raise NotImplementedError


class ClassificationLearner(Learner):

    def __init__(self, model, optimizer, crit, train_loader, val_loader=None, grad_clip=None, load_checkpoint=False,
                 name=''):
        super(ClassificationLearner, self).__init__(model, optimizer, crit, train_loader, val_loader, grad_clip,
                                                    load_checkpoint, name)
        self.train_accuracy = []
        self.val_accuracy = []

    def train_epoch(self, device, verbose=1):
        accuracies = []
        losses = []
        for batch, (data, labels) in tqdm(enumerate(self.train_loader)):
            data = data.to(device)
            labels = labels.to(device)
            y_pred = self.model.forward(data)
            loss = self.crit(y_pred, labels)
            self.backward(loss)
            if verbose == 1:
                losses += [loss.item()]
                accuracies += [(y_pred.max(dim=1)[1] == labels)]
        loss = np.mean(losses)
        self.losses += [loss]
        if verbose == 1:
            print('train loss: {:.4f} - train accuracy: {}\n'.format(loss, torch.cat(accuracies).float().mean().item()))

    def predict(self, data, device, prob=False):
        with torch.no_grad():
            y_pred = self.model.forward(data, device=device, train=False).to('cpu')
            if prob:
                y_pred = y_pred.data
            else:
                y_pred = torch.max(y_pred.data, 1)[1].data
            return y_pred

    def validate(self, device, verbose=0):
        with torch.no_grad():
            loss = []
            accuracies = []
            for data, y in self.val_loader:
                y_pred = self.model(data, device=device, train=False).to('cpu')
                loss += [self.crit(y_pred, y)]

                if verbose == 1:
                    y_pred = y_pred.max(dim=1)[1]
                    accuracies += [(y_pred == y).float()]

            loss = torch.stack(loss).mean().item()
            accuracy = torch.cat(accuracies).mean().item()
            self.val_accuracy += [accuracy]
            if verbose == 1:
                print('val loss: {:.4f} - val accuracy: {:.4f}'.format(loss, accuracy))
            return loss


