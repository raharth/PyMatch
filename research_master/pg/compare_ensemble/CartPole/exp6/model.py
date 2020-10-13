import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, in_nodes, out_nodes):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(in_nodes, 10)
        self.do1 = nn.Dropout(.5)
        self.fc2 = nn.Linear(10, out_nodes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.do1(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x
