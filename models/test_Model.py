import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, n_classes):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 28 * 28, 500)
        self.bn3 = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(500, n_classes)
        self.ankered_layers = [self.fc1, self.fc2]

    def forward(self, X, train=True, device='cpu'):
        if train:
            self.train()
        else:
            self.eval()

        x = X.to(device)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x.reshape(X.shape[0], -1)
        x = F.relu(x)
        x = self.fc1(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.fc2(x)
        out = F.softmax(x, dim=1)
        return out