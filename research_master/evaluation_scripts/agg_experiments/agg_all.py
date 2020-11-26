import torch
import matplotlib.pyplot as plt
import numpy as np


def compute_metrics(rewards):
    r_mean = rewards.mean(0)
    r_med = np.median(rewards, axis=0)
    r_std = rewards.std(0)
    return r_mean, r_std, r_med


def load_results(prefix, postfix, n_models, keys=[]):
    rewards = []
    for i in range(n_models):
        learner_dict = torch.load(f'{prefix}{i}{postfix}')
        for key in keys:
            learner_dict = learner_dict[key]
        rewards += [learner_dict]
    return np.array(rewards)


def add_metric2plot(rewards, color, label):
    r_mean, r_std, r_med = compute_metrics(rewards)
    plt.plot(r_mean, alpha=.5, label=label, c=color)
    plt.fill_between(np.arange(len(r_mean)), y1=r_mean - r_std, y2=r_mean + r_std, color=color, alpha=.1,
                     # label=r'$\sigma$-conf interval'
                     )
    return r_mean, r_std, r_med


mc_rewards = load_results('research_master/DQN/CartPole/mc_dropout/exp_72/exp_72_',
                          '/checkpoint/checkpoint_qlearner_mc.mdl',
                          n_models=20,
                          keys=['train_dict', 'val_reward'])

boosting_rewards = load_results('research_master/DQN/CartPole/boosting/exp_73/exp_73_',
                                '/checkpoint/checkpoint_ensemble.mdl',
                                n_models=15,
                                keys=['val_reward_mean'])

ensemble_rewards = load_results('research_master/DQN/CartPole/ensemble/exp_74/exp_74_',
                                '/checkpoint/checkpoint_ensemble.mdl',
                                n_models=10,
                                keys=['val_reward_mean'])


plt.title('Evaluation aggregation')
mc_r_mean, mc_r_std, mc_r_med = add_metric2plot(mc_rewards, color='green', label='MC')
boosting_r_mean, boosting_r_std, boosting_r_med = add_metric2plot(boosting_rewards, color='blue', label='Boosting')
ensemble_r_mean, ensemble_r_std, ensemble_r_med = add_metric2plot(ensemble_rewards, color='orange', label='Ensemble')

plt.legend()
plt.savefig('research_master/evaluation_scripts/agg_experiments/eval_agg.png')
plt.show()
