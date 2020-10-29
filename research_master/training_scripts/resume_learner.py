import sys
from pymatch.utils.experiment import Experiment
from pymatch.utils.functional import interactive_python_mode


if interactive_python_mode():
    path_scipt = 'research_master/pg/policy_gradient.py'
    root = 'research_master/pg/compare_ensemble/LunarLander/exp22'
else:
    path_scipt = sys.argv[0]
    root = sys.argv[1]

experiment = Experiment(root=root)
factory = experiment.get_factory()
params = experiment.get_params()
params['factory_args']['learner_args']['dump_path'] = root
Model = experiment.get_model_class()

factory_args = params['factory_args']
learner = factory(Model=Model, **factory_args)

learner.load_checkpoint(path=f'{root}/checkpoint', tag='checkpoint')
params['fit']['epochs'] = params['fit']['epochs'] - learner.train_dict['epochs_run']
learner.fit(**params['fit'])
# learner.resume_training(params['n_epochs'], 'cpu', restore_early_stopping=False, verbose=False)