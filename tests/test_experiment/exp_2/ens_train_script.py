import sys
import os
import numpy as np
from pymatch.ReinforcementLearning.memory import Memory, PriorityMemory
import pymatch.DeepLearning.hat as hat
import pymatch.DeepLearning.callback as cb
import pymatch.ReinforcementLearning.callback as rcb
import pymatch.ReinforcementLearning.selection_policy as sp
from pymatch.ReinforcementLearning.torch_gym import TorchGym
from pymatch.utils.experiment import Experiment, with_experiment
from pymatch.utils.functional import interactive_python_mode
from pymatch.DeepLearning.ensemble import DQNEnsemble
from pymatch.ReinforcementLearning.player import DQNPlayer, DQNPlayerCertainty


def get_selection_strategy(key, params):
    if key == 'AdaptiveQSelection':
        return sp.AdaptiveQActionSelectionEntropy(post_pipeline=[hat.EntropyHat()], **params)
    if key == 'QSelectionCertainty':
        return sp.QActionSelectionCertainty(post_pipeline=[hat.EntropyHat()], **params)
    if key == 'QSelection':
        return sp.QActionSelection(post_pipeline=[hat.EnsembleHat()], **params)
    if key == 'EpsilonGreedy':
        return sp.EpsilonGreedyActionSelection(post_pipeline=[hat.EnsembleHat()], **params)
    if key == 'AdaptiveEpsilonGreedy':
        return sp.AdaptiveEpsilonGreedyActionSelection(post_pipeline=[hat.EntropyHat()], **params)
    if key == 'Greedy':
        return sp.GreedyValueSelection(post_pipeline=[hat.EnsembleHat()], **params)
    raise ValueError('Unknown selection strategy')


def get_memory(key, params):
    if key == 'Memory':
        return Memory(**params)
    if key == 'PriorityMemory':
        return PriorityMemory(**params)
    raise ValueError('Unknown memory type')


def get_player(key, params={}):
    if key == 'DQN':
        return DQNPlayer(**params)
    if key == 'DQNCertainty':
        return DQNPlayerCertainty(**params)


def run(root, path_script):
    experiment = Experiment(root=root)
    factory = experiment.get_factory()
    params = experiment.get_params()
    # callbacks_factory = experiment.get_factory(source_function='get_ens_callbacks')
    params['factory_args']['learner_args']['dump_path'] = root
    # params['factory_args']['dump_path'] = root

    Model = experiment.get_model_class()
    experiment.document_script(path_script, overwrite=params['overwrite'])
    env = TorchGym(**params['env_args'])
    params['factory_args']['model_args']['in_nodes'] = env.env.observation_space.shape[0]
    params['factory_args']['model_args']['out_nodes'] = env.action_space.n
    params['factory_args']['env'] = env

    dqn_player = get_player(key=params['player_type'])
    selection_strategy = get_selection_strategy(params['selection_strategy'], params.get('selection_args', {}))

    with with_experiment(experiment=experiment, overwrite=params['overwrite']):
        memory = get_memory(params['memory_type'], params['memory_args'])
        params['factory_args']['learner_args']['memory'] = memory

        learner = DQNEnsemble(model_class=Model,
                              trainer_factory=factory,
                              memory=memory,
                              env=env,
                              player=dqn_player,
                              selection_strategy=selection_strategy,
                              trainer_args=params['factory_args'],
                              n_model=params['n_learner'],
                              dump_path=root,
                              callbacks=[
                                  rcb.EpisodeUpdater(**params.get('memory_update', {})),
                                  rcb.UncertaintyUpdater(hat=hat.EntropyHat()),
                                  cb.Checkpointer(frequency=10),
                                  rcb.EnvironmentEvaluator(
                                      env=env,
                                      n_evaluations=10,
                                      action_selector=sp.GreedyValueSelection(
                                          post_pipeline=[hat.EnsembleHat()]
                                      ),
                                      metrics={'det_val_reward_mean': np.mean, 'deter_val_reward_std': np.std},
                                      frequency=10,
                                      epoch_name='det_val_epoch'
                                  ),
                                  rcb.EnvironmentEvaluator(
                                      env=env,
                                      n_evaluations=10,
                                      action_selector=get_selection_strategy(params['eval_selection_strategy'],
                                                                             params.get('selection_args', {})),
                                      metrics={'prob_val_reward_mean': np.mean, 'prob_val_reward_std': np.std},
                                      frequency=10,
                                      epoch_name='prob_val_epoch'
                                  ),
                                  rcb.EnsembleRewardPlotter(frequency=10,
                                                            metrics={
                                                                'det_val_reward_mean': 'det_val_epoch',
                                                                'prob_val_reward_mean': 'prob_val_epoch',
                                                            })]
                              )
        learner.fit(**params['fit'])


if __name__ == '__main__':
    sys.path.append(os.path.abspath('..'))
    if interactive_python_mode():
        path_script = 'research_master/pg/policy_gradient.py'
        root = 'research_master/pg/compare_ensemble/LunarLander/exp22'
    else:
        path_script = sys.argv[0]
        root = sys.argv[1]
    run(root, path_script)