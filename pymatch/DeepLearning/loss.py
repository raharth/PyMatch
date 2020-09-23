from torch.nn.modules.loss import _Loss
import torch
import torch.nn as nn
from torch.distributions import normal
import pymatch.utils.functional as F

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
    # @todo what am I needing this for?
    def __init__(self, n_classes):
        super(OneHotBCELoss, self).__init__()
        self.n_classes = n_classes
        self.bce_loss = nn.BCELoss()

    def forward(self, input, target):
        target_onehot = torch.zeros((len(target), self.n_classes))
        target_onehot.scatter_(1, target.view(-1, 1), 1)
        return self.bce_loss(input, target_onehot)


class DistributionEstimationLoss(_Loss):
    def __init__(self):
        """
        This loss is used for a simple Regression task, where the output variable is assumed to be normal distributed.
        A brief explanation can be found here:
        https://www.inovex.de/blog/uncertainty-quantification-deep-learning/

        The model is assumed to have two output parameters instead of a single one. The first defines the expected
        value, while the second is the standard deviation.
        """
        super().__init__()

    def forward(self, input, target, reduce='mean'):
        """

        Args:
            input:      is expected to look like [[mu, sigma**2]]
            target:     target variable
            reduce:     reduce method, either `mean` or `sum`

        Returns:
            loss as torch tensor
        """
        mean = input[:, 0]
        std = input[:, 1]
        loss = (target - mean)**2/(2. * std) + std.log()/2.
        if reduce == 'mean':
            return loss.mean()
        if reduce == 'sum':
            return loss.sum()
        raise ValueError(f"reduce has to be either set as `mean` or `sum`, but was set as {reduce}")


class BrierLoss(_Loss):
    def __init__(self, weights=None, n_classes=None):
        """
        The Brier Loss is the MSE over the probability distribution for classification, assuming the target as a one-hot
        encoding.
        """
        super().__init__()
        if weights is None:
            self.mse = torch.nn.MSELoss()
        else:
            self.mse = WeightedMSE(weights=weights)
        self.n_classes = n_classes

    def forward(self, input, target):
        if self.n_classes is None:
            n_classes = input.shape[-1]
        else:
            n_classes = self.n_classes
        # @todo use one_hot_encoding function from F
        target_onehot = torch.zeros((len(target), n_classes)).to(target.device)
        target_onehot.scatter_(1, target.view(-1, 1), 1)
        return self.mse(input, target_onehot)


class WeightedMSE(_Loss):
    def __init__(self, weights, one_hot_targets=False, reduce='mean'):
        super().__init__()
        self.weights = weights
        self.one_hot_targets = one_hot_targets
        if reduce not in ['sum', 'mean']:
            raise ValueError(f'Unknown reduce method: {reduce}. Valid reductions are: `sum` and `mean`')
        self.reduce = reduce

    def forward(self, input, target):
        loss = ((input - target) ** 2) * self.weights
        if self.reduce == 'mean':
            return torch.mean(loss)
        return torch.sum(loss)
