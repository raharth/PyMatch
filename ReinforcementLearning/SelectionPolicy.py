import torch
import torch.nn.functional as F
from torch.distributions import Categorical


class SelectionPolicy:
    # @todo really necessary?
    def choose(self, q_values):
        raise NotImplementedError


class Softmax_Selection(SelectionPolicy):

    def __init__(self, temperature=1.):
        super(Softmax_Selection, self).__init__()
        self.temperature = temperature

    def choose(self, q_values):
        p = F.softmax(q_values / self.temperature, dim=1)
        dist = Categorical(p.squeeze())
        return dist.sample()


class EpsilonGreedy(SelectionPolicy):

    def __init__(self, epsilon):
        super(EpsilonGreedy, self).__init__()
        self.epsilon = epsilon

    def choose(self, q_values):
        if torch.rand(1) < self.epsilon:
            return torch.LongTensor(q_values.shape[0]).random_(0, q_values.shape[1])
        else:
            return q_values.argmax(dim=1)
