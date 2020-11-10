import sys
from pymatch.utils.experiment import Experiment
from pymatch.utils.functional import interactive_python_mode


if interactive_python_mode():
    path_scipt = 'research_master/training_scripts/train_A3C.py'
    root = 'research_master/A3C/CartPole/learner/exp_69'
else:
    path_scipt = sys.argv[0]
    root = sys.argv[1]

experiment = Experiment(root=root)
factory = experiment.get_factory()
critics_factory = experiment.get_factory('critics_factory')
params = experiment.get_params()
params['factory_args']['learner_args']['dump_path'] = root
Actor_Model = experiment.get_model_class(source_class='ActorModel')
Critics_Model = experiment.get_model_class(source_class='CriticsModel')
experiment.document_script(path_scipt, overwrite=params['overwrite'])
experiment.start(overwrite=params['overwrite'])

factory_args = params['factory_args']

critics = critics_factory(Model=Critics_Model, **params['crit_factory_args'])
learner = factory(Model=Actor_Model, critics=critics, **factory_args)

# learner.load_checkpoint(path=f'{root}/checkpoint')
learner.fit(**params['fit'])
# learner.resume_training(params['n_epochs'], 'cpu', restore_early_stopping=False, verbose=False)
experiment.finish()
