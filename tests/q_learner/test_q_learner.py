import torch
import numpy as np
from pymatch.ReinforcementLearning.loss import REINFORCELoss
from pymatch.ReinforcementLearning.memory import MemoryUpdater
from models.DQN1 import Model
from pymatch.ReinforcementLearning.torch_gym import TorchGym
import pymatch.ReinforcementLearning.callback as cb
import pymatch.ReinforcementLearning.policy_gradient as pg

torch.autograd.set_detect_anomaly(True)

model = Model(8, 4)
# env = TorchGym('CartPole-v1', max_episode_length=5000)
env = TorchGym('LunarLander-v2')
optim = torch.optim.SGD(model.parameters(), lr=.001, momentum=.5)
crit = torch.nn.MSELoss()
memory_updater = MemoryUpdater(memory_refresh_rate=.1)

learner = pg.QLearner(env=env,
                      model=model,
                      optimizer=optim,
                      memory_updater=memory_updater,
                      crit=crit,
                      action_selector=pg.QActionSelection(temperature=2.),
                      # action_selector=pg.EpsilonGreedyActionSelection(action_space=np.arange(env.action_space.n),
                      #                                                 epsilon=.95),
                      gamma=.95,
                      alpha=.1,
                      batch_size=256,
                      n_samples=8000,
                      grad_clip=20.,
                      memory_size=1000,
                      load_checkpoint=False,
                      name='test_Q',
                      callbacks=[cb.LastRewardPlotter(),
                                 cb.RewardPlotter(),
                                 cb.SmoothedRewardPlotter(window=6),
                                 cb.EnvironmentEvaluator(env=env, render=True, frequency=5)],
                      dump_path='tests/q_learner/tmp',
                      device='cpu')

learner.fit(100, 'cpu', restore_early_stopping=False, verbose=False)

# learner.load_checkpoint(learner.early_stopping_path)
# learner.train(10, 'cpu', checkpoint_int=100, render=True, restore_early_stopping=False, verbose=False)
#
# plt.plot(sliding_mean(learner.rewards, 50))
# plt.show()
#
# plt.plot(learner.rewards)
# plt.show()
