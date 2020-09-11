import torch
from pymatch.DeepLearning.callback import Callback
from pymatch.utils.functional import sliding_window
from pymatch.ReinforcementLearning.memory import Memory
import matplotlib.pyplot as plt



# class MemoryUpdater(Callback):
#     def __init__(self, memory_refresh_rate, update_frequ=1):
#         super().__init__()
#         if not 0. <= memory_refresh_rate <= 1.:
#             raise ValueError(f'memory_refresh_rate was set to {memory_refresh_rate} but has to be in ]0., 1.]')
#         self.memory_refresh_rate = memory_refresh_rate
#         self.update_frequ = update_frequ
#
#     def __call__(self, agent):
#         if agent.train_dict['epochs_run'] % self.update_frequ == 0:
#             reduce_to = int(len(agent.memory) * (1 - self.memory_refresh_rate))
#             self.memory.reduce_buffer(reduce_to)
#             self.fill_memory()
#
#     def fill_memory(self, agent):
#         while len(agent.memory) < agent.memory.buffer_size:
#             game = self.play_episode(agent=agent)
#             agent.memory.memorize(game)
#         agent.memory.reduce_buffer()
#
#     def play_episode(self, agent):
#         observation = agent.env.reset().detach()
#         episode_reward = 0
#         step_counter = 0
#         terminate = False
#         episode_memory = Memory(['log_prob', 'reward'])
#
#         while not terminate:
#             step_counter += 1
#             action, log_prob = agent.chose_action(observation)
#             new_observation, reward, done, _ = self.env.step(action)
#
#             episode_reward += reward
#             episode_memory.memorize((log_prob, torch.tensor(reward)), ['log_prob', 'reward'])
#             observation = new_observation
#             terminate = done or (self.max_episode_length is not None and step_counter >= self.max_episode_length)
#
#             # self.env.render()
#             if done:
#                 break
#
#         episode_memory.cumul_reward(gamma=self.gamma)
#         agent.memory.memorize(episode_memory, episode_memory.memory_cell_names)
#         agent.train_dict['rewards'] = agent.train_dict.get('rewards', []) + [episode_reward]
#         return episode_reward


class LastRewardPlotter(Callback):
    def __init__(self, frequency=1):
        super().__init__()
        self.frequency = frequency

    def __call__(self, model):
        if model.train_dict['epochs_run'] % self.frequency == 0:
            # plt.plot(model.train_dict['epochs_run'], model.train_dict['avg_reward'])
            plt.plot(model.train_dict['avg_reward'])
            plt.ylabel('average reward of last update')
            plt.xlabel('epochs/updates')
            plt.title('Average rewards per memory update')
            plt.tight_layout()
            plt.savefig(f'{model.dump_path}/last_rewards.png')
            plt.close()


class RewardPlotter(Callback):
    def __init__(self, frequency=1):
        super().__init__()
        self.frequency = frequency

    def __call__(self, model):
        if model.train_dict['epochs_run'] % self.frequency == 0:
            plt.plot(model.train_dict['rewards'])
            plt.ylabel('rewards')
            plt.xlabel('runs')
            plt.title('Average rewards per memory update')
            plt.tight_layout()
            plt.savefig(f'{model.dump_path}/rewards.png')
            plt.close()


class SmoothedRewardPlotter(Callback):
    def __init__(self, frequency=1, window=10):
        super().__init__()
        self.frequency = frequency
        self.window = window

    def __call__(self, model):
        if model.train_dict['epochs_run'] % self.frequency == 0 and \
                len(model.train_dict['rewards']) >= self.window:
            plt.plot(*sliding_window(self.window, model.train_dict['rewards']))
            plt.ylabel('rewards')
            plt.xlabel('epochs/updates')
            plt.title('Smoothed Average rewards per memory update')
            plt.tight_layout()
            plt.savefig(f'{model.dump_path}/smoothed_rewards.png')
            plt.close()
