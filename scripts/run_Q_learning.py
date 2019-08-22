import gym
import torch
import numpy as np
import matplotlib.pyplot as plt

from ReinforcementLearning.Q_Learner import Q_Learner
from models.PG1 import Model
from ReinforcementLearning.TorchEnv import TorchEnv
from ReinforcementLearning.SelectionPolicy import Softmax_Selection

from my_utils import sliding_mean


model = Model(4, 2)
env = TorchEnv('CartPole-v1')
selection_policy = Softmax_Selection(.1)
optim = torch.optim.SGD(model.parameters(), lr=.0001, momentum=.9)

learner = Q_Learner(agent=model,
                    optimizer=optim,
                    env=env,
                    selection_policy=selection_policy,
                    grad_clip=10.,
                    load_checkpoint=False)

learner.train(10000, 'cpu', checkpoint_int=100, render=False, verbose=False)

plt.plot(sliding_mean(learner.rewards, 50))
plt.show()
