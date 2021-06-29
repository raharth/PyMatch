import sys
import os
import numpy as np
from scipy.stats import entropy
import torch

from pymatch.ReinforcementLearning.memory import PriorityMemory
from pymatch.DeepLearning.hat import EnsembleHatStd, EnsembleHat
import pymatch.DeepLearning.callback as cb
import pymatch.ReinforcementLearning.callback as rcb
import pymatch.ReinforcementLearning.selection_policy as sp
from pymatch.ReinforcementLearning.torch_gym import TorchGym
from pymatch.utils.experiment import Experiment, with_experiment
from pymatch.utils.functional import interactive_python_mode, one_hot_encoding
from pymatch.DeepLearning.ensemble import DQNEnsemble
from pymatch.ReinforcementLearning.player import DQNPlayerCertainty


root = 'tests/test_experiment/exp_1'
experiment = Experiment(root=root)
factory = experiment.get_factory()
params = experiment.get_params()
params['factory_args']['learner_args']['dump_path'] = root

Model = experiment.get_model_class()
env = TorchGym(**params['factory_args']['env_args'])
params['factory_args']['model_args']['in_nodes'] = env.env.observation_space.shape[0]
params['factory_args']['model_args']['out_nodes'] = env.action_space.n

dqn_player = DQNPlayerCertainty()
selection_strategy = sp.AdaptiveQActionSelection(temperature=params['temp'],
                                                 min_length=10,
                                                 post_pipeline=[EnsembleHatStd()])

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


learner.fit(**params['fit'])

pred = learner(memory.memory['state'])
pred.shape
actions = pred.max(-1)[1]
ohe_actions = one_hot_encoding(actions, n_categories=4, unsqueeze=True)
action_probs = ohe_actions.mean(0)
action_entropy = entropy(torch.transpose(action_probs, 0, 1))
pred[:,0]