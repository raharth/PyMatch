import sys

from pymatch.DeepLearning.hat import EnsembleHat
import pymatch.ReinforcementLearning.callback as cb
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

core = experiment.get_model_class(source_file='core', source_class='Core')(**params['core_args'])
params['factory_args']['core'] = core

experiment.document_script(path_scipt, overwrite=params['overwrite'])
experiment.start(overwrite=params['overwrite'])


learner = Ensemble(model_class=Model,
                   trainer_factory=factory,
                   trainer_args=params['factory_args'],
                   n_model=params['n_learner'],
                   callbacks=[cb.EnvironmentEvaluator(
                       env=CartPole(**params['factory_args']['env_args']),
                       n_evaluations=10,
                       action_selector=GreedyValueSelection(
                           post_pipeline=[EnsembleHat()]
                       )),
                       cb.EnsembleRewardPlotter()])

for param_group in learner.learners[0].optimizer.param_groups:
    param_group['lr'] = param_group['lr'] * 10

# learner.load_checkpoint(f'{root}/checkpoint')
learner.fit(**params['fit'])
# learner.resume_training(params['n_epochs'], 'cpu', restore_early_stopping=False, verbose=False)
experiment.finish()