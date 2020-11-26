import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, in_nodes, out_nodes):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(in_nodes, 30)
        self.do1 = nn.Dropout(.2)
        self.fc2 = nn.Linear(30, 30)
        self.do2 = nn.Dropout(.2)
        self.fc3 = nn.Linear(30, out_nodes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.do1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.do2(x)
        x = self.fc3(x)
        return x

    def eval(self):
        super().train(False)
        self.do1.train()
        self.do2.train()