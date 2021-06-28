import sys
import os
import numpy as np
from pymatch.ReinforcementLearning.memory import PriorityMemory
from pymatch.DeepLearning.hat import EnsembleHatStd, EnsembleHat
import pymatch.DeepLearning.callback as cb
import pymatch.ReinforcementLearning.callback as rcb
import pymatch.ReinforcementLearning.selection_policy as sp
from pymatch.ReinforcementLearning.torch_gym import TorchGym
from pymatch.utils.experiment import Experiment, with_experiment
from pymatch.utils.functional import interactive_python_mode
from pymatch.DeepLearning.ensemble import DQNEnsemble
from pymatch.ReinforcementLearning.player import DQNPlayerCertainty


def run(root, path_script):
    print(root, path_script)
    experiment = Experiment(root=root)
    factory = experiment.get_factory()
    params = experiment.get_params()
    params['factory_args']['learner_args']['dump_path'] = root

    Model = experiment.get_model_class()
    experiment.document_script(path_script, overwrite=params['overwrite'])
    env = TorchGym(**params['factory_args']['env_args'])
    params['factory_args']['model_args']['in_nodes'] = env.env.observation_space.shape[0]
    params['factory_args']['model_args']['out_nodes'] = env.action_space.n

    dqn_player = DQNPlayerCertainty()
    selection_strategy = sp.AdaptiveQActionSelection(temperature=params['temp'],
                                                     min_length=10,
                                                     post_pipeline=[EnsembleHatStd()])

    with with_experiment(experiment=experiment, overwrite=params['overwrite']):
        memory = PriorityMemory(**params["factory_args"]['memory_args'])
        params['factory_args']['learner_args']['memory'] = memory

        learner = DQNEnsemble(model_class=Model,
                              trainer_factory=factory,
                              memory=memory,
                              env=env,
                              player=dqn_player,
                              selection_strategy=selection_strategy,
                              trainer_args=params['factory_args'],
                              n_model=params['n_learner'],
                              callbacks=[
                                  rcb.EpisodeUpdater(**params.get('memory_update', {})),
                                  cb.Checkpointer(frequency=1),
                                  rcb.UncertaintyUpdater(),
                                  rcb.EnvironmentEvaluator(
                                      env=TorchGym(**params['factory_args']['env_args']),
                                      n_evaluations=10,
                                      action_selector=sp.GreedyValueSelection(
                                          post_pipeline=[EnsembleHat()]
                                      ),
                                      metrics={'det_val_reward_mean': np.mean, 'deter_val_reward_std': np.std},
                                      frequency=1,
                                      epoch_name='det_val_epoch'
                                  ),
                                  rcb.EnvironmentEvaluator(
                                      env=TorchGym(**params['factory_args']['env_args']),
                                      n_evaluations=10,
                                      action_selector=sp.QActionSelection(temperature=params['temp'], post_pipeline=[EnsembleHat()]),
                                      metrics={'prob_val_reward_mean': np.mean, 'prob_val_reward_std': np.std},
                                      frequency=1,
                                      epoch_name='prob_val_epoch'
                                  ),
                                  rcb.EnsembleRewardPlotter(metrics={
                                      'det_val_reward_mean': 'det_val_epoch',
                                      'prob_val_reward_mean': 'prob_val_epoch',
                                  }),

                              ])

        # learner.load_checkpoint(path=f'{root}/checkpoint', tag='checkpoint')
        learner.fit(**params['fit'])
    # learner.resume_training(params['n_epochs'], 'cpu', restore_early_stopping=False, verbose=False)


if __name__ == '__main__':
    sys.path.append(os.path.abspath('..'))
    if interactive_python_mode():
        path_script = 'research_master/pg/policy_gradient.py'
        root = 'research_master/pg/compare_ensemble/LunarLander/exp22'
    else:
        path_script = sys.argv[0]
        root = sys.argv[1]
    run(root, path_script)
