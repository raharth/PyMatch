import torch
import matplotlib.pyplot as plt

from pytorch_lib.ReinforcementLearning.Q_Learner import Q_Learner
from models.DQN1 import Model
from pytorch_lib.ReinforcementLearning.TorchGym import TorchGym
from pytorch_lib.ReinforcementLearning.SelectionPolicy import EpsilonGreedy

from my_utils import sliding_mean


model = Model(4, 2)
env = TorchGym('CartPole-v1')
selection_policy = EpsilonGreedy(.05)
optim = torch.optim.SGD(model.parameters(), lr=.001, momentum=.5)

learner = Q_Learner(agent=model,
                    optimizer=optim,
                    env=env,
                    selection_policy=selection_policy,
                    grad_clip=10.,
                    load_checkpoint=False)

learner.train(10000, 'cpu', checkpoint_int=100, render=False, verbose=False)

plt.plot(sliding_mean(learner.rewards, 50))
plt.show()
