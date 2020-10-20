import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, core, out_nodes):
        super(Model, self).__init__()
        self.core = core
        self.fc1 = nn.Linear(24, out_nodes)

    def forward(self, x):
        x = self.core(x)
        x = self.fc1(x)
        x = F.softmax(x, dim=1)
        return x
