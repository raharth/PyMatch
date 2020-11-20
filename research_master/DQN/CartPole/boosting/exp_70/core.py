import torch
import torch.nn as nn
import torch.nn.functional as F


class Core(nn.Module):

    def __init__(self, in_nodes):
        super(Core, self).__init__()
        self.fc1 = nn.Linear(in_nodes, 24)
        self.fc2 = nn.Linear(24, 24)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return x