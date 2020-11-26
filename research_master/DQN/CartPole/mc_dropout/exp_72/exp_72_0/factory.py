import torch
import numpy as np
from pymatch.ReinforcementLearning.loss import REINFORCELoss
from pymatch.ReinforcementLearning.memory import MemoryUpdater
from models.DQN1 import Model
from pymatch.ReinforcementLearning.torch_gym import TorchGym, CartPole
import pymatch.ReinforcementLearning.callback as rcb
import pymatch.DeepLearning.callback as cb
import pymatch.ReinforcementLearning.learner as pg
import pymatch.ReinforcementLearning.selection_policy as sp
import pymatch.DeepLearning.hat as hat


def factory(Model, model_args, env_args, optim_args, memory_args, learner_args, n_learner):
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
                       action_selector=sp.QActionSelection(temperature=.3),
                       callbacks=[
                           cb.Checkpointer(),
                           rcb.EnvironmentEvaluator(env=env,
                                                    n_evaluations=10,
                                                    frequency=1,
                                                    action_selector=sp.GreedyValueSelection(
                                                        pre_pipeline=[hat.InputRepeater(n_learner)],
                                                        post_pipeline=[hat.EnsembleHat()]
                                                    )),
                           cb.MetricPlotter(frequency=1, metric='rewards', smoothing_window=100),
                           cb.MetricPlotter(frequency=1, metric='train_losses', smoothing_window=100),
                           cb.MetricPlotter(frequency=1, metric='avg_reward', smoothing_window=5),
                           cb.MetricPlotter(frequency=5, metric='val_reward', x='val_epoch', smoothing_window=5),
                       ],
                       **learner_args)
