from pymatch.ReinforcementLearning.memory import Memory
from pymatch.utils.functional import eval_mode
import numpy as np
import torch


class DQNPlayer:
    def __call__(self, agent, selection_strategy):
        observation = agent.env.reset().detach()
        episode_reward = 0
        step_counter = 0
        terminate = False
        episode_memory = Memory(['action', 'state', 'reward', 'new_state', 'terminal'],
                                gamma=agent.memory.gamma)
        with eval_mode(agent):
            while not terminate:
                step_counter += 1
                action = selection_strategy(agent, observation)
                new_observation, reward, terminate, _ = agent.env.step(action)

                episode_reward += reward
                episode_memory.memorize((action,
                                         observation,
                                         torch.tensor(reward).float(),
                                         new_observation,
                                         terminate),
                                        ['action', 'state', 'reward', 'new_state', 'terminal'])
                observation = new_observation

        agent.memory.memorize(episode_memory, episode_memory.memory_cell_names)
        agent.train_dict['rewards'] = agent.train_dict.get('rewards', []) + [episode_reward]

        if episode_reward > agent.train_dict.get('best_performance', -np.inf):
            agent.train_dict['best_performance'] = episode_reward

        return episode_reward