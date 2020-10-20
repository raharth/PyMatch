import sys
from pymatch.utils.experiment import Experiment
from pymatch.utils.functional import interactive_python_mode
import numpy as np
import matplotlib.pyplot as plt


if interactive_python_mode():
    path_scipt = 'research_master/pg/policy_gradient.py'
    root = 'research_master/pg/exp1'
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

val_rewards = []
# for k in range(params['n_learners']):
#     learner = factory(Model=Model, **params['factory_args'])
#     learner.fit(params['n_epochs'], 'cpu', restore_early_stopping=False, verbose=False)
#     val_rewards += [learner.train_dict['val_reward']]
from pymatch.DeepLearning.ensemble import Ensemble
learner = Ensemble(model_class=Model,
                   trainer_factory=factory,
                   trainer_args=params['factory_args'],
                   n_model=params['n_learner'])
learner.fit(params['n_epochs'], 'cpu', restore_early_stopping=False, verbose=False)
experiment.finish()
val_rewards = np.array(val_rewards)

plt.title(f'average validation reward over time for {params["n_learners"]} Agents')
plt.xlabel('epochs')
plt.ylabel('average reward')
plt.plot(val_rewards.mean(0))
for v in [val_rewards]:
    plt.plot(v, alpha=.5, color='grey')
plt.tight_layout()
plt.savefig(f'{root}/avg_val_rewards')