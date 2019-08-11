import torch
import torch.nn as nn

import os

class ReinforcementLearner:

    def __init__(self, agent, optimizer, env, crit, grad_clip=0., load_checkpoint=False, checkpoint_path=None,
                 early_stopping_path=None):
        self.agent = agent
        self.optimizer = optimizer
        self.crit = crit
        self.env = env

        self.losses = []
        self.epochs_run = 0
        self.best_performance = torch.tensor(float('inf'))

        self.grad_clip = grad_clip
        if (checkpoint_path is None or early_stopping_path is None)and not os.path.isdir('./tmp'):
            os.mkdir('./tmp')
        self.checkpoint_path = checkpoint_path if checkpoint_path is not None else './tmp/checkpoint.pth'
        self.early_stopping_path = early_stopping_path if early_stopping_path is not None else './tmp/early_stopping.pth'

        if load_checkpoint:
            self.load_checkpoint(self.checkpoint_path)

    def train(self, epochs, device, checkpoint_int=10, validation_int=10, restore_early_stopping=False):
        self.agent.train()

        for epoch in range(epochs):
            print('epoch: {}'.format(epoch))
            self.train_epoch(device)
            if epoch % checkpoint_int == 0:
                self.dump_checkpoint(self.epochs_run + epoch)
            if epoch % validation_int == 0:
                performance = self.validate(device=device)
                self.val_losses += [[performance, self.epochs_run + epoch]]
                if performance < self.best_performance:
                    self.best_performance = performance
                    self.dump_checkpoint(epoch=self.epochs_run + epoch, path=self.early_stopping_path)

        self.epochs_run += epochs
        if restore_early_stopping:
            self.load_checkpoint(self.early_stopping_path)
        self.dump_checkpoint(self.epochs_run, self.checkpoint_path)
        return

    def backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0.:
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.grad_clip)
        self.optimizer.step()
        return

    def dump_checkpoint(self, epoch, path=None):
        if path is None:
            path = self.checkpoint_path
        torch.save({'epoch': epoch,
                    'model_state_dict': self.agent.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': self.losses,
                    'accuracy': self.accuracy},
                   path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.agent.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epochs_run = checkpoint['epoch']
        self.losses = checkpoint['loss']
        self.accuracy = checkpoint['accuracy']
