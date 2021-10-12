import torch
import pymatch.ReinforcementLearning.learner as rl


def factory(Model, model_args, optim_args, learner_args, crit_args, name, env):
    model = Model(**model_args)

    optim = torch.optim.SGD(model.parameters(), **optim_args)
    crit = torch.nn.MSELoss(**crit_args)

    l_args = dict(learner_args)
    l_args['name'] = f"{learner_args['name']}_{name}"

    return rl.QLearner(env=env,
                       model=model,
                       optimizer=optim,
                       crit=crit,
                       action_selector=None,
                       fitter=rl.DQNFitter(),
                       **l_args)