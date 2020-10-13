import torch
import numpy as np
from pymatch.ReinforcementLearning.loss import REINFORCELoss
from pymatch.ReinforcementLearning.memory import MemoryUpdater
from models.DQN1 import Model
from pymatch.ReinforcementLearning.torch_gym import TorchGym, CartPole
import pymatch.ReinforcementLearning.callback as rcb
import pymatch.DeepLearning.callback as cb
import pymatch.ReinforcementLearning.learner as pg


def factory(Model, model_args, env_args, optim_args, memory_args, learner_args, name):
    model = Model(**model_args)
    env = CartPole(**env_args)
    memory_updater = MemoryUpdater(**memory_args)

    optim = torch.optim.SGD(model.parameters(), **optim_args)
    crit = torch.nn.MSELoss()

    return pg.QLearner(env=env,
                       model=model,
                       optimizer=optim,
                       memory_updater=memory_updater,
                       crit=crit,
                       action_selector=pg.QActionSelection(temperature=.3),
                       # action_selector=pg.EpsilonGreedyActionSelection(action_space=np.arange(env.action_space.n),
                       #                                                 epsilon=.95),
                       callbacks=[
                           rcb.EnvironmentEvaluator(env=env, n_evaluations=10, frequency=5),
                           # rcb.AgentVisualizer(env=env, frequency=5),
                           cb.MetricPlotter(frequency=1, metric='rewards', smoothing_window=100),
                           cb.MetricPlotter(frequency=1, metric='train_losses', smoothing_window=100),
                           cb.MetricPlotter(frequency=1, metric='avg_reward', smoothing_window=5),
                           cb.MetricPlotter(frequency=5, metric='val_reward', x='val_epoch', smoothing_window=5),
                       ],
                       dump_path='tests/q_learner/tmp',
                       device='cpu',
                       **learner_args)
