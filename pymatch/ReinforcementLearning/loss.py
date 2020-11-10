import torch
import torch.nn as nn


class REINFORCELoss(nn.Module):
    def __init__(self):
        super(REINFORCELoss, self).__init__()

    def forward(self, log_prob, rewards, baseline=None):
        if baseline is None:
            baseline = rewards.mean()
        return -(log_prob * rewards - baseline).sum()


class REINFORCELoss_moving_average(nn.Module):
    def __init__(self):
        super(REINFORCELoss_moving_average, self).__init__()
        self.m_avg = torch.zeros(1)
        self.m_avg2 = torch.zeros(1)
        self.count = torch.zeros(1)

    def forward(self, log_prob, rewards):
        # @todo this should be ok, but double check the computation
        self.m_avg = (self.m_avg + rewards.mean() / self.count).item()
        self.count += 1

        advantage = (rewards - self.m_avg).detach()
        return -(log_prob * advantage).sum()


class DQNLoss(nn.Module):
    """
    Loss according to 'Asynchronous Methods for Deep Reinforcement Learning' by Mnih et al.
    Strangely, this is not properly learning though...
    """
    def __init__(self):
        super().__init__()

    def forward(self, gamma, pred, max_next, reward):
        return (reward + gamma * max_next - pred).mean() ** 2
