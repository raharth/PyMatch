import torch.nn as nn
import torch.nn.functional as F


class LeakyHardTanH(nn.Module):
    def __init__(self, min_val=-1., max_val=1., leak=.1):
        super(LeakyHardTanH, self).__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.leak = leak

    def forward(self, x):
        y = ((F.leaky_relu(x - self.min_val, self.leak)) + self.min_val) * (x < 0.).float()
        z = (-(F.leaky_relu(- x + self.max_val, self.leak)) + self.max_val) * (x > 0.).float()
        return z + y