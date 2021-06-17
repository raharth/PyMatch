import torch
from pymatch.ReinforcementLearning.torch_gym import CartPole
import pymatch.ReinforcementLearning.callback as rcb
import pymatch.ReinforcementLearning.learner as pg
import pymatch.ReinforcementLearning.selection_policy as sp


def factory(Model, model_args, env_args, optim_args, memory_args, learner_args, name):
    model = Model(**model_args)
    env = CartPole(**env_args)

    optim = torch.optim.SGD(model.parameters(), **optim_args)
    crit = torch.nn.MSELoss()
    l_args = dict(learner_args)
    l_args['name'] = f"{learner_args['name']}_{name}"

    return pg.QLearner(env=env,
                       model=model,
                       optimizer=optim,
                       crit=crit,
                       action_selector=sp.QActionSelection(temperature=.3),
                       callbacks=[
                           rcb.MemoryUpdater(1.)
                       ],
                       **l_args)
