import gym
import torch
import numpy as np
import matplotlib.pyplot as plt

from ReinforcementLearning.PolicyGradient import PolicyGradient
from ReinforcementLearning.Loss import REINFORCELoss
from models.PG1 import Model
from ReinforcementLearning.TorchEnv import TorchEnv

model = Model(4, 2)
env = TorchEnv('CartPole-v1')
optim = torch.optim.SGD(model.parameters(), lr=.001, momentum=.7)
crit = REINFORCELoss()

learner = PolicyGradient(agent=model,
                         optimizer=optim,
                         env=env,
                         crit=crit,
                         grad_clip=20.)

learner.train(1, 'cpu', render=True)

plt.plot(learner.rewards)
plt.show()
