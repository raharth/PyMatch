from torch.nn.modules.loss import _Loss
import torch
import torch.nn as nn
from torch.distributions import normal


class L2Loss(_Loss):

    def __init__(self, crit, model, C):
        super(L2Loss, self).__init__()
        self.crit = crit
        self.model = model
        self.C = C

    def forward(self, input, target):
        loss = self.crit(input, target)
        device = loss.get_device() if loss.is_cuda else 'cpu'
        l2_reg = torch.tensor(0., device=device)
        for param in self.model.parameters():
            l2_reg = l2_reg + torch.norm(param)
        loss += self.C * l2_reg
        return loss


class AnkerLossClassification(_Loss):

    def __init__(self, crit, model, device, H):
        super(AnkerLossClassification, self).__init__()
        self.crit = crit
        self.model = model
        self.C = 1 / (2 * H)**0.5
        self.anker = []
        for layer in model.ankered_layers:
            params = layer._parameters['weight']
            m = normal.Normal(.0, H)
            self.anker += [m.sample(sample_shape=params.shape).to(device)]

    def forward(self, input, target):
        loss = self.crit(input, target)
        device = loss.get_device() if loss.is_cuda else 'cpu'
        l2_reg = torch.tensor(0., device=device)
        for layer, anker in zip(self.model.ankered_layers, self.anker):
            param = layer._parameters['weight']
            l2_reg = l2_reg + torch.norm(param - anker)
        loss = loss + self.C * l2_reg / len(target)   # C / N * L_2
        return loss


class OneHotBCELoss(_Loss):
    # @todo something is off here. Multiclass BCE is probably not gonna work
    def __init__(self, n_classes):
        super(OneHotBCELoss, self).__init__()
        self.n_classes = n_classes
        self.bce_loss = nn.BCELoss()

    def forward(self, input, target):
        target_onehot = torch.zeros((len(target), self.n_classes))
        target_onehot.scatter_(1, target.view(-1, 1), 1)
        return self.bce_loss(input, target_onehot)

