import torch.nn as nn
import torch.nn.functional as F


class CriticsModel(nn.Module):

    def __init__(self, in_nodes, out_nodes):
        super(CriticsModel, self).__init__()
        self.fc1 = nn.Linear(in_nodes, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, out_nodes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

class ActorModel(nn.Module):

    def __init__(self, in_nodes, out_nodes):
        super(ActorModel, self).__init__()
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