import torch
import numpy as np
from pymatch.ReinforcementLearning.memory import Memory, PriorityMemory
import pymatch.DeepLearning.hat as hat
import pymatch.ReinforcementLearning.selection_policy as sp
from pymatch.ReinforcementLearning.torch_gym import TorchGym
from pymatch.utils.experiment import Experiment
from pymatch.ReinforcementLearning.player import DQNPlayer, DQNPlayerCertainty
import pymatch.ReinforcementLearning.learner as rl
import pymatch.ReinforcementLearning.callback as rcb
import pymatch.DeepLearning.callback as cb


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


root = "tests/test_experiment/simple_DQN_integrated"

experiment = Experiment(root=root)
factory = experiment.get_factory()
params = experiment.get_params()
params['factory_args']['learner_args']['dump_path'] = root

Model = experiment.get_model_class()
env = TorchGym(**params['env_args'])
params['factory_args']['model_args']['in_nodes'] = env.env.observation_space.shape[0]
params['factory_args']['model_args']['out_nodes'] = env.action_space.n

dqn_player = get_player(key=params['player_type'])
selection_strategy = get_selection_strategy(params['selection_strategy'], params.get('selection_args', {}))

memory = get_memory(params['memory_type'], params['memory_args'])
params['factory_args']['learner_args']['memory'] = memory
params['factory_args']['learner_args']['selection_strategy'] = selection_strategy
# params['factory_args']['learner_args']['player'] = dqn_player
model = Model(**params['factory_args']['model_args'])

optim = torch.optim.SGD(model.parameters(), **params['factory_args']['optim_args'])
crit = torch.nn.MSELoss(**params['factory_args']['crit_args'])

learner = rl.QLearner(model=model, optimizer=optim, crit=crit, env=env,
                      player=dqn_player,
                      callbacks=[
                          # rcb.EpisodeUpdater(**params.get('memory_update', {})),
                          cb.Checkpointer(frequency=10),
                          rcb.EnvironmentEvaluator(
                              env=env,
                              n_evaluations=10,
                              action_selector=sp.GreedyValueSelection(),
                              metrics={'det_val_reward_mean': np.mean},
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
                          cb.MetricPlotter()
                      ],
                      fitter=rl.DQNIntegratedFitter(sample_size=1024),
                      **params['factory_args']['learner_args'])

learner.fit(epochs=params['fit']['epochs'],
            device=params['fit']['device'],
            verbose=params['fit']['verbose'])
