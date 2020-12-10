import torch
import torch.nn as nn
import torch.nn.functional as F

from pymatch.utils.experiment import Experiment, with_experiment
from pymatch.DeepLearning.ensemble import Ensemble
from pymatch.ReinforcementLearning.loss import REINFORCELoss
from pymatch.ReinforcementLearning.memory import MemoryUpdater
from pymatch.ReinforcementLearning.torch_gym import CartPole
import pymatch.ReinforcementLearning.learner as rl
import pymatch.ReinforcementLearning.selection_policy as sp


class Core(nn.Module):

    def __init__(self, in_nodes):
        super(Core, self).__init__()
        self.fc1 = nn.Linear(in_nodes, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        return x


class Model(nn.Module):

    def __init__(self, core, out_nodes):
        super(Model, self).__init__()
        self.core = core
        self.do1 = nn.Dropout(.5)
        self.fc2 = nn.Linear(10, out_nodes)

    def forward(self, x):
        x = self.core(x)
        x = self.do1(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x


def factory(Model, core, model_args, env_args, optim_args, memory_args, learner_args, name):
    model = Model(core, **model_args)
    env = CartPole(**env_args)
    optim = torch.optim.SGD(model.parameters(), **optim_args)
    crit = REINFORCELoss()
    memory_updater = MemoryUpdater(**memory_args)
    largs = dict(learner_args)
    largs['name'] = f"{learner_args['name']}_{name}"
    learner = rl.PolicyGradient(env=env,
                                model=model,
                                optimizer=optim,
                                memory_updater=memory_updater,
                                crit=crit,
                                action_selector=sp.PolicyGradientActionSelection(),
                                callbacks=[],
                                **largs
                                )
    return learner


root = 'tests/experiment'

experiment = Experiment(root=root)
params = experiment.get_params()
params['factory_args']['learner_args']['dump_path'] = root
params['factory_args']['core'] = Core(**params['core_args'])

with with_experiment(experiment=experiment, overwrite=params['overwrite']):
    learner = Ensemble(model_class=Model,
                       trainer_factory=factory,
                       trainer_args=params['factory_args'],
                       n_model=params['n_learner'],
                       callbacks=[])
    learner.fit(**params['fit'])
    print('done training')
    # raise NotImplemented
