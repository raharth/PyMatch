import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from my_utils import sliding_mean

from ReinforcementLearning.PolicyGradient import PolicyGradient
from ReinforcementLearning.Loss import REINFORCELoss
from models.PG1 import Model
from ReinforcementLearning.TorchEnv import TorchEnv

model = Model(4, 2)
env = TorchEnv('CartPole-v1')
optim = torch.optim.SGD(model.parameters(), lr=.001, momentum=.1)
crit = REINFORCELoss()

learner = PolicyGradient(agent=model,
                         optimizer=optim,
                         env=env,
                         crit=crit,
                         grad_clip=20.,
                         load_checkpoint=False)

learner.train(1000, 'cpu', checkpoint_int=100, render=False, restore_early_stopping=True, verbose=False)

plt.plot(learner.rewards)
plt.show()

plt.plot(sliding_mean(learner.rewards, 50))
plt.show()