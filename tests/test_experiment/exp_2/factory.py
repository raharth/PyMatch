import torch
import numpy as np
from pymatch.ReinforcementLearning.torch_gym import TorchGym
import pymatch.ReinforcementLearning.callback as rcb
import pymatch.DeepLearning.callback as cb
import pymatch.ReinforcementLearning.learner as rl
import pymatch.DeepLearning.hat as hat
import pymatch.ReinforcementLearning.selection_policy as sp
from pymatch.ReinforcementLearning.memory import Memory


def factory(Model, model_args, optim_args, learner_args, crit_args, name, env):
    model = Model(**model_args)
    # env = TorchGym(**env_args)

    optim = torch.optim.SGD(model.parameters(), **optim_args)
    crit = torch.nn.MSELoss(**crit_args)

    l_args = dict(learner_args)
    l_args['name'] = f"{learner_args['name']}_{name}"

    return rl.QLearner(env=env,
                       model=model,
                       optimizer=optim,
                       crit=crit,
                       action_selector=None,
                       # memory=Memory(**memory_args),
                       callbacks=[
                           # rcb.EnvironmentEvaluator(env=env,
                           #                          n_evaluations=10,
                           #                          frequency=10,
                           #                          action_selector=sp.GreedyValueSelection()),
                       ],
                       **l_args)


def get_ens_callbacks(params, env):
    return [
        rcb.EpisodeUpdater(**params.get('memory_update', {})),
        cb.Checkpointer(frequency=10),
        rcb.UncertaintyUpdater(head=hat.EntropyHat()),
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