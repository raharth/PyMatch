from pymatch.ReinforcementLearning.callback import UncertaintyUpdater
from pymatch.ReinforcementLearning.learner import ReinforcementLearner
import numpy as np
import torch


class MockLearner(ReinforcementLearner):
    def __init__(self, memory):
        self.train_loader = memory
        self.train_dict = {'epochs_run': 0}
        self.device = 'cpu'

    def __call__(self, X):
        return X


class MockMemory:
    def __init__(self, memory: dict):
        self.memory = memory

    def sample_loader(self, shuffle):
        return [[[0], torch.tensor(self.memory['state']), [0], [0], [0]]]


class MockHat:
    def __call__(self, X, *args, **kwargs):
        std = torch.zeros(size=(len(X), 4))
        std[:, 0] = torch.tensor(np.arange(10, 20))
        return X, std


updater = UncertaintyUpdater(head=MockHat())
learner = MockLearner(memory=MockMemory({'state': np.arange(0, 10).reshape(-1, 1, 1)}))

updater(learner)

learner.train_loader.memory


torch.load
