import torch
import numpy as np
from pymatch.ReinforcementLearning.loss import REINFORCELoss
from pymatch.ReinforcementLearning.memory import MemoryUpdater
from models.DQN1 import Model
from pymatch.ReinforcementLearning.torch_gym import TorchGym, CartPole
import pymatch.ReinforcementLearning.callback as rcb
import pymatch.DeepLearning.callback as cb
import pymatch.ReinforcementLearning.learner as pg

torch.autograd.set_detect_anomaly(True)

model = Model(4, 2)
env = CartPole()
# env = TorchGym('LunarLander-v2')
optim = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
crit = torch.nn.MSELoss()
memory_updater = MemoryUpdater(memory_refresh_rate=.1)

learner = pg.QLearner(model=model, optimizer=optim, crit=crit, env=env,
                      selection_strategy=pg.QActionSelection(temperature=.3), alpha=.2, gamma=.95, player=,
                      memory_size=10000, n_samples=8000, batch_size=256, grad_clip=5., name='test_Q', callbacks=[
        rcb.EnvironmentEvaluator(env=env, n_evaluations=10, frequency=5),
        # rcb.AgentVisualizer(env=env, frequency=5),
        cb.MetricPlotter(frequency=1, metric='rewards', smoothing_window=100),
        cb.MetricPlotter(frequency=1, metric='train_losses', smoothing_window=100),
        cb.MetricPlotter(frequency=1, metric='avg_reward', smoothing_window=5),
        cb.MetricPlotter(frequency=5, metric='val_reward', x='val_epoch', smoothing_window=5),
    ], dump_path='tests/q_learner/tmp', device='cpu', memory_updater=memory_updater, load_checkpoint=False)

learner.fit(30, 'cpu', restore_early_stopping=False, verbose=False)

env.close()
