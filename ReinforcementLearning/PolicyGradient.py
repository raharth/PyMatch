# general imports
import numpy as np
from tqdm import tqdm

# torch imports
import torch
import torch.nn as nn
from torch.distributions import Categorical

# own imports
from ReinforcementLearning.ReinforcementLearner import ReinforcementLearner


class PolicyGradient(ReinforcementLearner):

    def __init__(self, agent, optimizer, env, crit, grad_clip=None, load_checkpoint=False):
        """

        Args:
            agent (nn.Module): neural network
            optimizer (torch.optim): Optimizer
            env(any): environment to interact with
            crit (any): loss function
        """
        super(PolicyGradient, self).__init__(agent, optimizer, env, crit, grad_clip=grad_clip, load_checkpoint=load_checkpoint)

    def train_epoch(self, device, verbose=1):
        # @todo check
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
        if verbose:
            print('accuracy: {:.4f} \n'.format(np.mean(accuracy)))
        return accuracy

    def validate(self, device, verbose=0):
        # @todo check
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

    def predict(self, data_loader, device, prob=False):
        # @todo check
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

    def chose_action(self, observation):
        probs = self.agent(observation)
        dist = Categorical(probs)
        return dist.sample()
