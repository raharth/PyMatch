import torch
import torch.nn as nn


class ReinforceMultivariant(nn.Module):
    def __init__(self, std=.1):
        super(ReinforceMultivariant, self).__init__()
        self.std = std

    def forward(self, actions):
        std = torch.eye(actions.shape[1]) * self.std**2
        dist = torch.distributions.MultivariateNormal(actions, std)
        action = dist.sample()
        return action, dist.log_prob(action)