import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, in_nodes, out_nodes):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(in_nodes, 50)
        self.do1 = nn.Dropout(.5)
        self.fc2 = nn.Linear(50, 50)
        self.do2 = nn.Dropout(.5)
        self.fc3 = nn.Linear(50, out_nodes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.do1(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.do2(x)

        x = self.fc3(x)
        x = F.softmax(x, dim=-1)
        return x

    def eval(self):
        super().train(False)
        self.do1.train()
        self.do2.train()
