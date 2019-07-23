import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np


class EnsembleTrainer:

    def __init__(self, model_factory, n_model, optim_factory, loss_crit, model_args={}, optim_args={}):
        self.models = [model_factory(**model_args) for _ in range(n_model)]
        self.optims = [optim_factory(**optim_args) for _ in range(n_model)]
        self.loss_crit = loss_crit
        self.epochs_run = 0

        self.losses = []

    def predict(self, x):
        y_preds = torch.stack([model(x) for model in self.models])
        y_pred_mean = y_preds.mean(dim=0)
        y_pred_std = y_preds.std(dim=0)
        return y_pred_mean, y_pred_std

    def train(self, epochs, device, checkpoint_int=10, validation_int=10, restore_early_stopping=False):
        # @todo
        for i in range(len(self.models)):
            self.train_model(epochs, device, checkpoint_int, validation_int, restore_early_stopping)

    def train_model(self, idx, epochs, device, checkpoint_int=10, validation_int=10, restore_early_stopping=False):
        # @todo
        model = self.models[idx]
        model.train()
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

    def train_epoch(self, device):
        # @todo
        accuracies = []
        for batch, (data, labels) in tqdm(enumerate(self.train_loader)):
            data = data.to(device)
            labels = labels.to(device)
            action_probs = self.agent.forward(data, device=device)

            actions, log_probs = self.agent.sample(action_probs, device=device)

            accuracy, rewards = self.get_rewards(actions, labels)
            accuracies += [accuracy]

            loss = self.crit(log_probs, rewards)

            self.losses += [loss]

            self.backward(loss)
        self.accuracy += [np.mean(accuracies)]
        print('accuracy: {:.4f} \n'.format(np.mean(accuracy)))
        return accuracy

    def dump_checkpoint(self, epochs, path=None):
        if path is None:
            path = self.checkpoint_path
        torch.save({'epoch': epochs,
                    'model_state_dict': [model.state_dict() for model in self.models],
                    'optimizer_state_dict': [optimizer.state_dict() for optimizer in self.optimizer],
                    'losses': self.losses,
                    'accuracy': self.accuracy,
                    'n_models': len(self.models)},
                   path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)

        for model, state in zip(self.models, checkpoint['model_state_dict']):
            model.load_state_dict(state)

        for optim, state in zip(self.optims, checkpoint['model_state_dict']):
            optim.load_state_dict(state)

        self.epochs_run = checkpoint['epochs']
        self.losses = checkpoint['loss']
