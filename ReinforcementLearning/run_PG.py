import gym
import torch
import numpy as np

from ReinforcementLearning.PolicyGradient import PolicyGradient
from ReinforcementLearning.Loss import REINFORCELoss
from models.PG1 import Model
from ReinforcementLearning.TorchEnv import TorchEnv

model = Model(4, 2)
env = TorchEnv('CartPole-v1')
optim = torch.optim.SGD(model.parameters(), lr=.01)
crit = REINFORCELoss()

learner = PolicyGradient(agent=model,
                       optimizer=optim,
                       env=env,
                       crit=crit)

learner.train(1, 'cpu')


env.step(0)


env.action_space