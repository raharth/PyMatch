import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, core, out_nodes):
        super(Model, self).__init__()
        self.core = core
        self.fc1 = nn.Linear(30, out_nodes)

    def forward(self, x):
        x = self.core(x)
        x = self.fc1(x)
        return x

    def parameters_module(self):
        core_params, head_params = [], []
        for np in self.named_parameters():
            if np[0].split('.')[0] == 'core':
                core_params += [np[1]]
            else:
                head_params += [np[1]]
        return core_params, head_params
