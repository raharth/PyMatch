import sys
import numpy as np
from pymatch.DeepLearning.hat import EnsembleHat
from pymatch.ReinforcementLearning.memory import Memory
import pymatch.DeepLearning.callback as cb
import pymatch.ReinforcementLearning.callback as rcb
from pymatch.ReinforcementLearning.selection_policy import GreedyValueSelection
from pymatch.ReinforcementLearning.torch_gym import TorchGym, CartPole
from pymatch.utils.experiment import Experiment
from pymatch.utils.functional import interactive_python_mode
from pymatch.DeepLearning.ensemble import Ensemble

if interactive_python_mode():
    path_scipt = 'research_master/train_boosting.py'
    root = 'research_master/pg/boosting/CartPole/exp36'
else:
    path_scipt = sys.argv[0]
    root = sys.argv[1]

experiment = Experiment(root=root)
factory = experiment.get_factory()
params = experiment.get_params()
params['factory_args']['learner_args']['dump_path'] = root
Model = experiment.get_model_class()

Core = experiment.get_model_class(source_file='core', source_class='Core')
params['factory_args']['core'] = Core(**params['core_args'])

experiment.document_script(path_scipt, overwrite=params['overwrite'])
experiment.start(overwrite=params['overwrite'])

memory = Memory(['action', 'state', 'reward', 'new_state'], **params['memory_args'])
params['factory_args']['learner_args']['memory'] = memory

learner = Ensemble(model_class=Model,
                   trainer_factory=factory,
                   trainer_args=params['factory_args'],
                   n_model=params['n_learner'],
                   callbacks=[
                       cb.Checkpointer(),
                       rcb.EnvironmentEvaluator(
                           env=CartPole(**params['factory_args']['env_args']),
                           n_evaluations=10,
                           action_selector=GreedyValueSelection(
                               post_pipeline=[EnsembleHat()]
                           ),
                           metrics={'val_reward_mean': np.mean, 'val_reward_std': np.std}
                       ),
                       rcb.EnsembleRewardPlotter(metrics={'val_reward_mean': 'val_epoch',
                                                          })])

learner.fit(**params['fit'])
experiment.finish()
