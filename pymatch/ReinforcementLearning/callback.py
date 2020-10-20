import torch
from pymatch.DeepLearning.callback import Callback
from pymatch.utils.functional import sliding_window
from pymatch.ReinforcementLearning.memory import Memory
import matplotlib.pyplot as plt
from pymatch.utils.functional import eval_mode
from pymatch.ReinforcementLearning.selection_policy import GreedyValueSelection
import numpy as np
from tqdm import tqdm


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


class SmoothedRewardPlotter(Callback):
    def __init__(self, frequency=1, window=10):
        super().__init__()
        self.frequency = frequency
        self.window = window

    def __call__(self, model):
        if model.train_dict['epochs_run'] % self.frequency == 0 and \
                len(model.train_dict['rewards']) >= self.window:
            plt.plot(*sliding_window(self.window, model.train_dict['rewards']))
            plt.ylabel('averaged rewards')
            plt.xlabel('episodes')
            plt.title('Smoothed Average rewards')
            plt.tight_layout()
            plt.savefig(f'{model.dump_path}/smoothed_rewards.png')
            plt.close()


class EnvironmentEvaluator(Callback):
    def __init__(self,
                 env,
                 frequency=1,
                 n_evaluations=1,
                 action_selector=GreedyValueSelection(),
                 metric_name='val_reward',
                 epoch_name='val_epoch'):
        super().__init__()
        self.env = env
        self.frequency = frequency
        self.action_selector = action_selector
        self.n_evaluations = n_evaluations
        self.metric_name = metric_name
        self.epoch_name = epoch_name

    def __call__(self, model):
        if model.train_dict['epochs_run'] % self.frequency == 0:
            print('Evaluation environment...')
            with eval_mode(model):
                episode_rewards = []
                for _ in tqdm(range(self.n_evaluations)):
                    terminate = False
                    episode_reward = 0
                    observation = self.env.reset().detach()
                    while not terminate:
                        action = self.action_selector(model, observation)
                        new_observation, reward, done, _ = self.env.step(action)
                        episode_reward += reward
                        observation = new_observation
                        terminate = done
                    episode_rewards += [episode_reward]

            print(f'Evaluation reward for {model.name}: {np.mean(episode_rewards):.2f}')
            model.train_dict[self.metric_name] = model.train_dict.get(self.metric_name, []) + [np.mean(episode_rewards)]
            model.train_dict[self.epoch_name] = model.train_dict.get(self.epoch_name, []) + [model.train_dict['epochs_run']]


class AgentVisualizer(Callback):
    def __init__(self, env, frequency=1, action_selector=GreedyValueSelection()):
        super().__init__()
        self.env = env
        self.frequency = frequency
        self.action_selector = action_selector

    def __call__(self, model):
        if model.train_dict['epochs_run'] % self.frequency == 0:
            print('Visualizing environment...')
            with torch.no_grad():
                with eval_mode(model):
                    terminate = False
                    episode_reward = 0
                    observation = self.env.reset().detach()
                    while not terminate:
                        action = self.action_selector(model, observation)
                        new_observation, reward, done, _ = self.env.step(action)
                        episode_reward += reward
                        observation = new_observation
                        terminate = done
                        self.env.render()

            print(f'Visual evaluation reward for model: {episode_reward:.2f}')


class EnsembleRewardPlotter(Callback):
    def __init__(self):
        super().__init__()

    def __call__(self, model):
        val_rewards = np.array([learner.train_dict['val_reward'] for learner in model.learners])
        plt.title(f'average validation reward over time for {len(model.learners)} Agents')
        plt.xlabel('epochs')
        plt.ylabel('average reward')
        plt.plot(val_rewards.mean(0), label='agent average')
        for v in val_rewards:
            plt.plot(v, alpha=.1, color='grey')
        plt.plot(model.train_dict['val_reward'], label='ensemble')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{model.dump_path}/avg_ensemble_val_rewards.png')
        plt.close()
