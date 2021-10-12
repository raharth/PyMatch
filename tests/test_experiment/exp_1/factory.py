import torch
import numpy as np
from pymatch.ReinforcementLearning.torch_gym import TorchGym
import pymatch.ReinforcementLearning.callback as rcb
import pymatch.DeepLearning.callback as cb
import pymatch.ReinforcementLearning.learner as rl
import pymatch.ReinforcementLearning.selection_policy as sp
from pymatch.ReinforcementLearning.memory import Memory


def factory(Model, model_args, env_args, optim_args, memory_args, learner_args, crit_args, temp, name):
    model = Model(**model_args)
    env = TorchGym(env_args['env_name'])

    optim = torch.optim.SGD(model.parameters(), **optim_args)
    crit = torch.nn.MSELoss(**crit_args)

    l_args = dict(learner_args)
    l_args['name'] = f"{learner_args['name']}_{name}"

    return rl.QLearner(env=env,
                       model=model,
                       optimizer=optim,
                       crit=crit,
                       fitter=rl.DQNFitter(),
                       action_selector=sp.QActionSelection(temperature=temp),
                       callbacks=[],
                       **l_args)