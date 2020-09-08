import torch
import torch.nn as nn


class REINFORCELoss(nn.Module):
    def __init__(self):
        super(REINFORCELoss, self).__init__()

    def forward(self, log_prob, rewards):
        advantage = (rewards - rewards.mean())
        return -(log_prob * advantage).sum()


class REINFORCELoss_moving_average(nn.Module):
    def __init__(self):
        super(REINFORCELoss, self).__init__()
        self.m_avg = torch.zeros(1)
        self.m_avg2 = torch.zeros(1)
        self.count = torch.zeros(1)

    def forward(self, log_prob, rewards):
        # @todo this should be ok, but double check the computation
        self.m_avg = self.m_avg + rewards.mean() / self.count
        self.count += 1

        advantage = (rewards - self.m_avg)
        return -(log_prob * advantage).sum()