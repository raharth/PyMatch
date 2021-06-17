import torch
from pymatch.ReinforcementLearning.learner import NormalThompsonSampling
from torch.distributions import Categorical, MultivariateNormal

y_m = torch.tensor([[.2, .5, .3],
                    [1.2, 1.5, 1.3]])
y_s = torch.tensor([[.1, .2, .1],
                    [.1, .2, .1]])
shape = y_m.shape
dist = MultivariateNormal(loc=y_m.view(-1),
                          covariance_matrix=y_s.view(-1) ** 2 * torch.eye(len(y_s.view(-1))))
dist.sample((1,)).view(shape)


class DummyAgent:
    def __init__(self, y_s):
        class DummyModel:
            def __init__(self):
                self.training = False

            def eval(self):
                pass

        self.model = DummyModel()
        self.y_s = y_s

    def __call__(self, x, device):
        return x, self.y_s

    def eval(self):
        pass


y_m = torch.tensor([[.2, .5, .3],
                    [1.2, 1.5, 1.3]])
y_s = torch.tensor([[.1, .2, .1],
                    [.1, .2, .1]])
dummy_agent = DummyAgent(y_s)
ts = NormalThompsonSampling(pre_pipes=[], post_pipes=[])

test = torch.stack([ts(dummy_agent, y_m) for k in range(100000)])
test_m = test.mean(0)
test_s = test.std(0)