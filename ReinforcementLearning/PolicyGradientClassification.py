# general imports
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time

# torch imports
import torch
import torch.nn as nn

plt.style.use('seaborn')


class PolicyGradientClassification:

    def __init__(self, agent, optimizer, train_loader, n_classes, crit, exclusion_reward=0., val_loader=None,
                 grad_clip=None, load_checkpoint=False):
        """
        Args:
            agent (nn.Module): neural network
            optimizer (torch.optim): Optimizer
            train_loader (torchvision.datasets): training data
            val_loader (torchvision.datasets): validation data
            n_classes (int): number of classes
            crit (any): loss function
            exclusion_reward (float ]-1,1[ ): reward for excluding a data point
        """
        self.agent = agent
        self.optimizer = optimizer
        self.crit = crit

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.n_classes = n_classes
        self.exclusion_reward = exclusion_reward

        self.grad_clip = grad_clip
        self.checkpoint_path = './reinforcement_learning_torch/RL_default_class/tmp/checkpoint.pth'
        self.early_stopping_path = './reinforcement_learning_torch/RL_default_class/tmp/early_stopping.pth'

        self.losses = []
        self.val_losses = []
        self.accuracy = []
        self.epochs_run = 0
        self.best_performance = np.inf

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

    def train_epoch(self, device):
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

    def get_rewards(self, actions, labels):
        rewards = (labels == actions).type(torch.float)
        rewards[actions == self.n_classes] = self.exclusion_reward

        # report only on predicted/non-excluded data points
        reported_rewards = rewards[actions != self.n_classes]
        accuracy = [reported_rewards.to('cpu').numpy().mean() if len(reported_rewards) > 0 else 0.]

        rewards = rewards * 2 - 1   # scale to [-1,1]
        return accuracy, rewards

    def backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.grad_clip)
        self.optimizer.step()

    def validate(self, device, verbose=0):
        self.agent.eval()

        with torch.no_grad():
            correct = 0.
            total = 0.
            for data, y in self.val_loader:
                action_probs = self.agent(data, device=device).to('cpu')
                _, predicted = torch.max(action_probs.data, 1)
                y = y[predicted != self.n_classes]
                predicted = predicted[predicted != self.n_classes]
                total += y.size(0)
                correct += (predicted == y).sum().item()
            if verbose == 1:
                print('accuracy: {:.4f}'.format(correct / total))
            return correct / total

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

    def predict(self, data_loader, device, prob=False):
        self.agent.eval()
        with torch.no_grad():
            predictions = []
            for batch, (data, _) in tqdm(enumerate(data_loader)):
                data = data.to(device)
                action_probs = self.agent.forward(data, device=device).to('cpu')
                if prob:
                    return action_probs.numpy()
                actions = torch.max(action_probs.data, 1)[1].numpy()
                predictions += [actions]
            return np.concatenate(predictions)