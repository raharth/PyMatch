import torch
import torch.nn as nn
import torch.nn.functional as F


# class Model(nn.Module):
#
#     def __init__(self, in_nodes, out_nodes):
#         super(Model, self).__init__()
#         self.fc1 = nn.Linear(in_nodes, 24)
#         self.fc2 = nn.Linear(24, 24)
#         self.fc3 = nn.Linear(24, out_nodes)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.fc2(x)
#         x = F.relu(x)
#         x = self.fc3(x)
#         return x

class Model(nn.Module):

    def __init__(self, in_nodes, out_nodes):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(in_nodes, 10)
        self.bn1 = nn.BatchNorm1d(10)
        self.do1 = nn.Dropout(.1)
        # self.fc2 = nn.Linear(10, 10)
        # self.do2 = nn.Dropout(.1)
        self.fc3 = nn.Linear(10, out_nodes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        # x = F.sigmoid(x)
        x = self.do1(x)

        # x = self.fc2(x)
        # x = F.relu(x)
        # x = self.do2(x)

        x = self.fc3(x)
        return x