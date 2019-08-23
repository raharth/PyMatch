import gym
import torch
import numpy as np
import matplotlib.pyplot as plt

from ReinforcementLearning.Q_Learner import Q_Learner
from models.DQN1 import Model
from ReinforcementLearning.TorchEnv import TorchEnv
from ReinforcementLearning.SelectionPolicy import Softmax_Selection, EpsilonGreedy

from my_utils import sliding_mean


model = Model(4, 2)
env = TorchEnv('CartPole-v1')
selection_policy = EpsilonGreedy(.1)
optim = torch.optim.SGD(model.parameters(), lr=.001, momentum=.0)

learner = Q_Learner(agent=model,
                    optimizer=optim,
                    env=env,
                    selection_policy=selection_policy,
                    grad_clip=-1.,
                    load_checkpoint=False)

learner.train(50000, 'cpu', checkpoint_int=100, render=False, verbose=False)

plt.plot(sliding_mean(learner.rewards, 50))
plt.show()
