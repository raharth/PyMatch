import torch
from pymatch.DeepLearning.callback import Callback
from pymatch.utils.functional import sliding_window
from pymatch.ReinforcementLearning.memory import Memory
import matplotlib.pyplot as plt
from pymatch.utils.functional import eval_mode
from pymatch.ReinforcementLearning.selection_policy import GreedyValueSelection
import numpy as np
from tqdm import tqdm
from pymatch.DeepLearning.pipeline import Pipeline
from pymatch.DeepLearning.hat import EnsembleHatStd



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

### DEPRICATED! THIS SHOULD BE REPLACED BY MetricPlotter
# class SmoothedRewardPlotter(Callback):
#     def __init__(self, window=10, metrics=None, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.window = window
#         self.metrics = metrics if metrics is not None else ['rewards']
#
#     def forward(self, model):
#         for metric in self.metrics:
#             if len(model.train_dict['rewards']) >= self.window:
#                 plt.plot(*sliding_window(self.window, model.train_dict['rewards']))
#         plt.ylabel('averaged rewards')
#         plt.xlabel('episodes')
#         plt.title('Smoothed Average rewards')
#         plt.tight_layout()
#         plt.savefig(f'{model.dump_path}/smoothed_rewards.png')
#         plt.close()


class EnvironmentEvaluator(Callback):
    def __init__(self,
                 env,
                 n_evaluations=1,
                 action_selector=GreedyValueSelection(),
                 metrics=None,
                 epoch_name='val_epoch',
                 *args,
                 **kwargs):
        """
        Evaluates an environment and writes the restults to the train_dict of the learner.

        Args:
            env:                Environment to evaluate
            frequency:          Every 'frequency'-th epoch the environment is evaluated.
            n_evaluations:      Number of evaluations to run at a time to obtain a more reliable estimate
            action_selector:    SelectionPolicy to use
            metrics:            dict, of the patter {name_to_store: fn_to apply to the n runs}. By default is uses the
                                mean, but it can be replaced with the median or the std could be added.
            epoch_name:         Defines under which name the epochs are stored, when evaluating the environment
        """
        super().__init__(*args, **kwargs)
        self.env = env
        self.action_selector = action_selector
        self.n_evaluations = n_evaluations
        self.metrics = {'val_reward': np.mean} if metrics is None else metrics
        self.epoch_name = epoch_name

    def forward(self, model):
        if model.train_dict['epochs_run'] % self.frequency == 0:
            print('Evaluation environment...', flush=True)
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

            print(f'Evaluation reward for {model.name}: {np.mean(episode_rewards):.2f}', flush=True)
            for name, func in self.metrics.items():
                model.train_dict[name] = model.train_dict.get(name, []) + [func(episode_rewards)]
            model.train_dict[self.epoch_name] = model.train_dict.get(self.epoch_name, []) + [model.train_dict['epochs_run']]


class AgentVisualizer(Callback):
    def __init__(self, env, action_selector=GreedyValueSelection(), *args, **kwargs):
        """
        Visualizes an agent in an environment.

        Args:
            env:                Environment to evaluate
            frequency:          Every 'frequency'-th epoch the environment is evaluated
            action_selector:    Selection Policy with which the agent is used
        """
        super().__init__()
        self.env = env
        self.action_selector = action_selector

    def forward(self, model):
        print('Visualizing environment...')
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
    def __init__(self, metrics=None, xlabel='epochs', ylabel='average reward',
                 title='average validation reward over time', *args, **kwargs):
        """
        This plots the individual learners of an ensemble as well as the aggregated performance metrics of the ensemble.

        Args:
            metrics:    dict of the name of metrics to plot for the ensemble. The key is the y-values, while the values
                        is the according x-values. This way metrics estimated at different epochs can be plotted
                        properly in the same plot.
        """
        super().__init__(*args, **kwargs)
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.metrics = {'val_reward_mean': 'val_epoch'} if metrics is None else metrics

    def forward(self, model):
        val_rewards = np.array([learner.train_dict.get('val_reward', []) for learner in model.learners])
        learner_val_epochs = model.learners[0].train_dict.get('val_epoch', [])
        plt.title(self.title)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.plot(learner_val_epochs, val_rewards.mean(0), label=f'mean ({len(model.learners)} agents)')
        plt.plot(learner_val_epochs, np.median(val_rewards, axis=0), label=f'median ({len(model.learners)} agents)')
        for v in val_rewards:
            plt.plot(learner_val_epochs, v, alpha=.1, color='grey')
        for y, x in self.metrics.items():
            plt.plot(model.train_dict[x], model.train_dict[y], label=f'ensemble {y}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{model.dump_path}/avg_ensemble_val_rewards.png')
        plt.close()


class MemoryUpdater(Callback):
    def __init__(self, memory_refresh_rate, *args, **kwargs):
        """
        Updates the memory of
        Args:
            memory_refresh_rate: fraction of oldest memories to be replaced when updated
        """
        super().__init__(*args, **kwargs)
        if not 0. <= memory_refresh_rate <= 1.:
            raise ValueError(f'memory_refresh_rate was set to {memory_refresh_rate} but has to be in ]0., 1.]')
        self.memory_refresh_rate = memory_refresh_rate

    def forward(self, agent):
        reduce_to = int(len(agent.train_loader) * (1 - self.memory_refresh_rate))
        agent.train_loader.reduce_buffer(reduce_to)
        self.fill_memory(agent)

    def fill_memory(self, agent):
        print('Filling Memory')
        reward, games = 0, 0
        while len(agent.train_loader) < agent.train_loader.memory_size:
            reward += agent.play_episode()
            games += 1
        agent.train_loader.reduce_buffer()
        agent.train_dict['avg_reward'] = agent.train_dict.get('avg_reward', []) + [reward / games]

    def start(self, agent):
        self.fill_memory(agent)


class EpisodeUpdater(Callback):
    """
    Sampels and writes a singe episode to the memory of an agent.
    """
    def forward(self, agent):
        reward = agent.play_episode()
        agent.train_loader.reduce_buffer()
        agent.train_dict['avg_reward'] = agent.train_dict.get('avg_reward', []) + [reward]

    def start(self, model):
        self.forward(model)


class SingleEpisodeSampler(Callback):
    """
    Always samples just a single episode from the environment.
    """
    def forward(self, agent):
        agent.train_loader.memory_reset()
        reward = agent.play_episode()
        agent.train_dict['avg_reward'] = agent.train_dict.get('avg_reward', []) + [reward]

    def start(self, model):
        self.forward(model)


class StateCertaintyEstimator(Callback):
    def forward(self, agent, *args, **kwargs):
        pipeline = Pipeline(pipes=[agent, EnsembleHatStd()])
        pred, certainty = pipeline(torch.cat(agent.train_loader.memory['state']))
        certainty = certainty.mean(-1)
        certainty /= certainty.sum()
        agent.train_loader.set_probabilities(certainty)
