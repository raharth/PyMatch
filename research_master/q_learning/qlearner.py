import sys

from pymatch.DeepLearning.hat import EnsembleHat
import pymatch.ReinforcementLearning.callback as cb
from pymatch.ReinforcementLearning.learner import GreedyValueSelection
from pymatch.ReinforcementLearning.torch_gym import TorchGym
from pymatch.utils.experiment import Experiment
from pymatch.utils.functional import interactive_python_mode
from pymatch.DeepLearning.ensemble import Ensemble
import numpy as np
import matplotlib.pyplot as plt


if interactive_python_mode():
    path_scipt = 'research_master/q_learning/qlearner.py'
    root = 'research_master/q_learning/exp_q_1'
else:
    path_scipt = sys.argv[0]
    root = sys.argv[1]

experiment = Experiment(root=root)
factory = experiment.get_factory()
params = experiment.get_params()
params['factory_args']['learner_args']['dump_path'] = root
Model = experiment.get_model_class()
experiment.document_script(path_scipt, overwrite=params['overwrite'])
experiment.start(overwrite=params['overwrite'])

factory_args = params['factory_args']
learner = factory(Model=Model,
                  model_args=factory_args['model_args'],
                  env_args=factory_args['env_args'],
                  optim_args=factory_args['optim_args'],
                  memory_args=factory_args['memory_args'],
                  learner_args=factory_args['learner_args'])

# learner.load_checkpoint(path=f'{root}/checkpoint')
learner.fit(**params['fit'])
# learner.resume_training(params['n_epochs'], 'cpu', restore_early_stopping=False, verbose=False)
experiment.finish()
