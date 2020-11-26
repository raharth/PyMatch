import torch
import matplotlib.pyplot as plt
import numpy as np

rewards = []
for i in range(20):
    path = f'research_master/DQN/CartPole/mc_dropout/exp_72_{i}/checkpoint/checkpoint_qlearner_mc.mdl'
    learner_dict = torch.load(path)
    rewards += [learner_dict['train_dict']['val_reward']]

rewards = np.array(rewards)


plt.title('MC Dropout aggregation')
r_mean = rewards.mean(0)
r_med = np.median(rewards, axis=0)
r_std = rewards.std(0)
# plt.plot(rewards.transpose(), c='grey', alpha=.1)
plt.plot(r_mean, alpha=.5, label='mean')
plt.plot(r_med, alpha=.5, label='median')
# plt.plot(r_mean + r_std, c='green', alpha=.1)
# plt.plot(r_mean - r_std, c='green', alpha=.1)
plt.fill_between(np.arange(len(r_mean)), y1=r_mean-r_std, y2=r_mean+r_std, color='green', alpha=.1, label=r'$\sigma$-conf interval')
plt.legend()
plt.show()