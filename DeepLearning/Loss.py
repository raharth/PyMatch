from torch.nn.modules.loss import _Loss
import torch


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
            l2_reg += torch.norm(param)
        loss += self.C * l2_reg
        return loss


class AnkerLoss(_Loss):

    def __init__(self, crit, model, C):
        super(AnkerLoss, self).__init__()
        self.crit = crit
        self.model = model
        self.C = C
        self.anker = [params.clone().detach() for params in model.parameters()]

    def forward(self, input, target):
        loss = self.crit(input, target)
        device = loss.get_device() if loss.is_cuda else 'cpu'
        l2_reg = torch.tensor(0., device=device)
        for param, anker in zip(self.model.parameters(), self.anker):
            l2_reg += torch.norm(param - anker)
        loss += self.C * l2_reg
        return loss


