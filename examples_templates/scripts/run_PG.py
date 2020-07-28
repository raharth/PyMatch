import torch
import matplotlib.pyplot as plt

from pytorch_lib.ReinforcementLearning.PolicyGradient import PolicyGradient
from pytorch_lib.ReinforcementLearning.Loss import REINFORCELoss
from models.PG1 import Model
from pytorch_lib.ReinforcementLearning.TorchGym import TorchGym

from my_utils import sliding_mean


model = Model(4, 2)
env = TorchGym('CartPole-v1')
optim = torch.optim.SGD(model.parameters(), lr=.0001, momentum=.1)
crit = REINFORCELoss()

learner = PolicyGradient(agent=model,
                         optimizer=optim,
                         env=env,
                         crit=crit,
                         grad_clip=20.,
                         load_checkpoint=False)

learner.train(2000, 'cpu', checkpoint_int=100, render=False, restore_early_stopping=False, verbose=False)
learner.load_checkpoint(learner.early_stopping_path)
learner.train(10, 'cpu', checkpoint_int=100, render=True, restore_early_stopping=False, verbose=False)

plt.plot(sliding_mean(learner.rewards, 50))
plt.show()

plt.plot(learner.rewards)
plt.show()