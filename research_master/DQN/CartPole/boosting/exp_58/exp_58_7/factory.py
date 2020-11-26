import torch
import numpy as np
from pymatch.ReinforcementLearning.loss import REINFORCELoss
from pymatch.ReinforcementLearning.memory import MemoryUpdater
from pymatch.ReinforcementLearning.torch_gym import TorchGym, CartPole
import pymatch.ReinforcementLearning.callback as rcb
import pymatch.DeepLearning.callback as cb
import pymatch.ReinforcementLearning.learner as pg
import pymatch.ReinforcementLearning.selection_policy as sp


def factory(Model, core, model_args, env_args, optim_args, memory_args, learner_args, name, n_learner):
    model = Model(core, **model_args)
    env = CartPole(**env_args)
    memory_updater = MemoryUpdater(**memory_args)

    core_params, head_params = model.parameters_module()

    optim = torch.optim.SGD([{'params': core_params,
                              'lr': optim_args['lr']/n_learner},
                             {'params': head_params}], **optim_args)
    crit = torch.nn.MSELoss()
    learner_args['name'] = name

    return pg.QLearner(env=env,
                       model=model,
                       optimizer=optim,
                       memory_updater=memory_updater,
                       crit=crit,
                       action_selector=sp.QActionSelection(temperature=.3),
                       callbacks=[
                           rcb.EnvironmentEvaluator(env=env, n_evaluations=10, frequency=1),
                       ],
                       **learner_args)

# for param_group in optim.param_groups:
#     print(param_group.keys())
#
# for i, p in enumerate(model.parameters()):
#     print(i)
#     if p is in list(core.parameters()):
#         print(i)
#
# list(model.parameters())[1] is list(core.parameters())[1]
#
# core_params = [np for np in model.named_parameters() if np[0].split('.')[0] == 'core']
# head_params = [np for np in model.named_parameters() if not np[0].split('.')[0] == 'core']


