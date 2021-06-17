import torch
import matplotlib.pyplot as plt

from pymatch.ReinforcementLearning.loss import REINFORCELoss
from pymatch.ReinforcementLearning.memory import MemoryUpdater
from models.PG1 import Model
from pymatch.ReinforcementLearning.torch_gym import TorchGym
import pymatch.ReinforcementLearning.callback as rcb
import pymatch.DeepLearning.callback as cb
import pymatch.ReinforcementLearning.learner as rl

from my_utils import sliding_mean

torch.autograd.set_detect_anomaly(True)
model = Model(4, 2)
env = TorchGym('CartPole-v1', max_episode_length=None)
# env = TorchGym('FrozenLake8x8-v0', max_episode_length=None)
optim = torch.optim.SGD(model.parameters(), lr=.0001, momentum=.8)
crit = REINFORCELoss()
memory_updater = MemoryUpdater(memory_refresh_rate=.1)

learner = rl.PolicyGradient(env=env,
                            model=model,
                            optimizer=optim,
                            memory_updater=memory_updater,
                            crit=crit,
                            action_selector=rl.PolicyGradientActionSelection(),
                            # action_selector=pg.BayesianDropoutActionSelection(50),
                            gamma=.9,
                            batch_size=256,
                            n_samples=2048,
                            grad_clip=20.,
                            memory_size=1000,
                            load_checkpoint=False,
                            name='test_pg',
                            # callbacks=[cb.LastRewardPlotter(),
                            #            cb.RewardPlotter(),
                            #            cb.SmoothedRewardPlotter(window=6),
                            #            cb.EnvironmentEvaluator(env=env, render=True, frequency=5)],
                            callbacks=[
                               rcb.EnvironmentEvaluator(env=env, n_evaluations=10, frequency=5),
                               rcb.AgentVisualizer(env=env, frequency=5),
                               cb.MetricPlotter(frequency=1, metric='rewards', smoothing_window=100),
                               cb.MetricPlotter(frequency=1, metric='train_losses', smoothing_window=100),
                               cb.MetricPlotter(frequency=1, metric='avg_reward', smoothing_window=5),
                               cb.MetricPlotter(frequency=5, metric='val_reward', x='val_epoch', smoothing_window=5),
                           ],
                            dump_path='tests/policy_gradient/tmp',
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
