import torch
from torch import nn
import numpy as np
from tqdm import tqdm


class ClassificationTrainer:

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
        self.name = name

        self.losses = []
        self.val_losses = []
        self.epochs_run = 0
        self.best_performance = np.inf

        if load_checkpoint:
            self.load_checkpoint(self.checkpoint_path)

    def train(self, epochs, device, checkpoint_int=10, validation_int=10, restore_early_stopping=False, verbose=1):
        self.model.train()
        for epoch in range(epochs):
            if verbose == 1:
                print('epoch: {}'.format(epoch))

            self.train_epoch(device)

            if epoch % checkpoint_int == 0:
                self.dump_checkpoint(self.epochs_run + epoch)

            if epoch % validation_int == 0:
                performance = self.validate(device=device)
                self.val_losses += [[performance, self.epochs_run + epoch]]
                if performance < self.best_performance:
                    self.best_performance = performance
                    self.dump_checkpoint(epoch=self.epochs_run + epoch, path=self.early_stopping_path,
                                         tag='early_stopping')

        self.epochs_run += epochs
        if restore_early_stopping:
            self.load_checkpoint(self.early_stopping_path, 'early_stopping')
        self.dump_checkpoint(self.epochs_run, self.checkpoint_path)

    def train_epoch(self, device, verbose=1):
        accuracies = []
        losses = []
        for batch, (data, labels) in tqdm(enumerate(self.train_loader)):
            labels = labels.to(device)
            y_pred = self.model.forward(data, device=device)
            loss = self.crit(y_pred, labels)
            self.losses += [loss.item()]
            self.backward(loss)
            if verbose == 1:
                losses += [loss.item()]
                accuracies += [(y_pred.max(dim=1)[1] == labels).item()]
        if verbose == 1:
            print('loss: {:.4f} - accuracy: {}\n'.format(np.mean(losses), np.mean(accuracies)))

    def backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

    def predict(self, data_loader, device, prob=False):
        self.model.eval()
        with torch.no_grad():
            predictions = []

            for batch, (data, _) in tqdm(enumerate(data_loader)):
                y_pred = self.model.forward(data, device=device).to('cpu')

                if prob:
                    y_pred = y_pred.numpy()
                else:
                    y_pred = torch.max(y_pred.data, 1)[1].numpy()
                predictions += [y_pred]
            return np.concatenate(predictions, axis=0)

    def validate(self, device, verbose=0):
        self.model.eval()
        with torch.no_grad():
            loss = []
            for data, y in self.val_loader:
                y_pred = self.model(data, device=device).to('cpu')
                loss += [self.crit(y_pred, y)]
            loss = torch.cat(loss).mean().item()
            if verbose == 1:
                print('accuracy: {:.4f}'.format(loss))
            return loss

    def dump_checkpoint(self, epoch, path=None, tag='checkpoint'):
        path = self.get_path(path=path, tag=tag)
        torch.save({'epoch': epoch,
                    'model_state_dict': self.agent.state_dict(),
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
