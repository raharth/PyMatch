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


def factory(Model, model_args, env_args, optim_args, memory_args, learner_args, name):
    model = Model(**model_args)
    env = CartPole(**env_args)
    memory_updater = MemoryUpdater(**memory_args)

    optim = torch.optim.SGD(model.parameters(), **optim_args)
    crit = torch.nn.MSELoss()
    learner_args['name'] = name

    return pg.QLearner(env=env,
                       model=model,
                       optimizer=optim,
                       memory_updater=memory_updater,
                       crit=crit,
                       action_selector=sp.QActionSelection(temperature=.3),
                       callbacks=[
                           cb.Checkpointer(),
                           rcb.EnvironmentEvaluator(env=env, n_evaluations=10, frequency=1),
                       ],
                       **learner_args)
