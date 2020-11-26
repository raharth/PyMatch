import torch
import torch.nn as nn
import torch.nn.functional as F


class Core(nn.Module):

    def __init__(self, in_nodes):
        super(Core, self).__init__()
        self.fc1 = nn.Linear(in_nodes, 30)
        self.do1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(30, 30)
        self.do2 = nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.do1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.do2(x)
        return x