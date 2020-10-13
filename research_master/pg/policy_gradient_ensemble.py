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
experiment.document_script(path_scipt, overwrite=params['overwrite'])
experiment.start(overwrite=params['overwrite'])


learner = Ensemble(model_class=Model,
                   trainer_factory=factory,
                   trainer_args=params['factory_args'],
                   n_model=params['n_learner'],
                   callbacks=[cb.EnvironmentEvaluator(
                       env=TorchGym(**params['factory_args']['env_args']),
                       n_evaluations=10,
                       action_selector=GreedyValueSelection(
                           post_pipeline=[EnsembleHat()]
                       )),
                       cb.EnsembleRewardPlotter()])
# learner.load_checkpoint(f'{root}/checkpoint')
learner.fit(**params['fit'])
# learner.resume_training(params['n_epochs'], 'cpu', restore_early_stopping=False, verbose=False)
experiment.finish()

# val_rewards = np.array([learner.train_dict['val_reward'] for learner in learner.learners])
# # val_rewards2 = val_rewards[:5, :]
# plt.title(f'average validation reward over time for {params["n_learner"]} Agents')
# plt.xlabel('epochs')
# plt.ylabel('average reward')
# plt.plot(val_rewards.mean(0), label='agent average')
# for v in val_rewards:
#     plt.plot(v, alpha=.1, color='grey')
# plt.plot(learner.train_dict['val_reward'], label='ensemble')
# plt.legend()
# plt.tight_layout()
# plt.savefig(f'{root}/avg_val_rewards')
# plt.show()
# plt.close()

viz = cb.AgentVisualizer(env=TorchGym(**params['factory_args']['env_args']),
                   frequency=1,
                   action_selector=GreedyValueSelection(
                       post_pipeline=[EnsembleHat()]
                   ))
learner.train_dict['epochs_run'] = 0
viz(learner)