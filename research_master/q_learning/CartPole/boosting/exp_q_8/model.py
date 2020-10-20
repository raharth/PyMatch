import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, core, out_nodes):
        super(Model, self).__init__()
        self.core = core
        self.do1 = nn.Dropout(.5)
        self.fc2 = nn.Linear(50, out_nodes)

    def forward(self, x):
        x = self.core(x)
        x = self.do1(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x
