import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch import Tensor


class FixableDropout(torch.nn.modules.dropout._DropoutNd):
    def __init__(self, p: float = .5, inplace: bool = False):
        super(FixableDropout, self).__init__(p=p, inplace=inplace)
        self.mask = None
        self.fixed = False

    def forward(self, input: Tensor) -> Tensor:
        if not self.fixed or self.mask is None:
            self.mask = torch.autograd.Variable(torch.bernoulli(input.data.new(input.data.size()).fill_(self.p)))
        if self.inplace:
            input *= self.mask
            return input
        else:
            return input * self.mask

    def fix(self, set_to: bool = True):
        self.fixed = set_to

    def unfix(self):
        self.fix(False)


class fix_dropout:
    def __init__(self, fixable_modules):
        self.fixable_modules = fixable_modules

    def __enter__(self):
        for fixable in self.fixable_modules:
            fixable.fix()

    def __exit__(self):
        for fixable in self.fixable_modules:
            fixable.unfix()


class LinearEnsemble(nn.Module):
    def __init__(self, in_features: int = 2, out_features: int = 3, nr_models: int = 4, bias: bool = True,
                 device=None, dtype=None):
        """
        I guess that was an attempt of having a Ensemble that is running in parallel using larger weight matrices.
        I guess I failed...
        Args:
            in_features:
            out_features:
            nr_models:
            bias:
            device:
            dtype:
        """
        super().__init__()
        self.nr_models = nr_models
        factory_kwargs = {'device': device, 'dtype': dtype}
        weight = torch.zeros((in_features * nr_models, out_features * nr_models))
        self.weight_list = []
        for i in range(nr_models):
            params = torch.normal(mean=torch.zeros(in_features, out_features))
            self.weight_list += [params]
            weight[i * in_features: (i + 1) * in_features, i * out_features: (i + 1) * out_features] = params
        self.weight = Parameter(weight)
        if bias:
            self.bias = Parameter(torch.empty(out_features * nr_models, **factory_kwargs))
        else:
            self.bias = None

    def forward(self, input: Tensor) -> Tensor:
        input = input.repeat(1, self.nr_models)
        output = torch.mm(input, self.weight)
        if self.bias is not None:
            output += self.bias
        return output

    def to(self, device):
        super(LinearEnsemble, self).to(device)
        self.weight = self.weight.to(device)
        self.weight_list = [weight.to(device) for weight in self.weight_list]

# le = LinearEnsemble(300, 400, 10, bias=False)
# le.to('cuda')
#
# le.weight
#
# x = torch.rand(size=(1, 200), device='cuda')
# import time
#
# t = time.time()
# for i in range(1000):
#     res1 = le(x)
#     # T += [time.time() - t]
# print(f'le time {time.time() - t}')
#
#
# t = time.time()
# for _ in range(1000):
#     for i in range(10):
#         torch.mm(x, le.weight_list[i])
# print(f'loop time {time.time() - t}')
#
# le.weight.to('cuda')
# le.weight
#
#
# x.repeat(1, 2)
#
#
#
#
#
# x = torch.rand(size=(1, 1, 5))
#
# conv1 = torch.nn.Conv1d(in_channels=1, out_channels=10, kernel_size=5)
# conv2 = torch.nn.Conv1d(in_channels=1, out_channels=10, kernel_size=5)
#
# conv1(x)
#
# x = torch.tensor([[1., 2, 1, 2]])
# m = torch.tensor([[[1, 2], [3, 4]], [[2, 4], [6, 8]]])
# r = torch.matmul(x, torch.transpose(m, 0, 1))
# r
#
# m = torch.zeros((4, 4), requires_grad=True)
# m[0:2, 0:2] = torch.tensor([[1, 2], [3, 4.]])
# m[-2:, -2:] = torch.tensor([[2, 4], [6, 8]])
# r = torch.matmul(x, m)
# r
#
# ms = m.to_sparse()
# xs = x.to_sparse()
# r = torch.matmul(x, m)
# ms.to_dense()
#
# (x.to_sparse()).addmm(ms)
#
# torch.sparse.mm(xs, ms).to_dense()
#
#
#
#
# nn.Linear(2, 3).weight
#
# F.linear(xs, ms)
#
# import matplotlib.pyplot as plt
#
# test = torch.rand((100000,))
# test = torch.normal(mean=torch.zeros(100000, ))
# test = test.numpy()
# plt.hist(test, bins=1000)
# plt.show()
#
# type(xs)