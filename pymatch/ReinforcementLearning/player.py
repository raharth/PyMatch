from pymatch.ReinforcementLearning.memory import Memory
from pymatch.utils.functional import eval_mode
import numpy as np
import torch


def get_player(key, params={}):
    if key == 'DQN':
        return DQNPlayer(**params)
    if key == 'DuelingDQN':
        return DuelingDQNPlayer(**params)
    if key == 'DQNCertainty':
        return DQNPlayerCertainty(**params)


class RLPlayer:
    def __init__(self):
        """
        A player determine how an episode is played by an RL agent.
        This is only a base class to inherent from
        """
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class DQNPlayer(RLPlayer):
    """
    This determines how a DQN plays a single episode, storing the trajectory in the agents memory.
    """
    def __call__(self, agent, selection_strategy, memory):
        observation = agent.env.reset().detach()
        episode_reward = 0
        step_counter = 0
        terminate = False
        episode_memory = Memory(['action', 'state', 'reward', 'new_state', 'terminal'],
                                gamma=memory.gamma)
        with eval_mode(agent):
            while not terminate:
                step_counter += 1
                agent.to(agent.device)
                action = selection_strategy(agent, observation.to(agent.device))
                new_observation, reward, terminate, _ = agent.env.step(action)

                episode_reward += torch.sum(reward).item() / agent.env.n_instances
                episode_memory.memorize((action,
                                         observation,
                                         torch.tensor(reward).float(),
                                         new_observation,
                                         terminate),
                                        ['action', 'state', 'reward', 'new_state', 'terminal'])
                observation = new_observation[~terminate.view(-1)]
                terminate = terminate.min().item()

        memory.memorize(episode_memory, episode_memory.memory_cell_names)
        agent.train_dict['rewards'] = agent.train_dict.get('rewards', []) + [episode_reward]
        agent.train_dict['env_steps'] = agent.train_dict.get('env_steps', 0) + 1

        if episode_reward > agent.train_dict.get('best_performance', -np.inf):
            agent.train_dict['best_performance'] = episode_reward

        return episode_reward


class DQNPlayerCertainty(RLPlayer):
    """
    This determines how a uncertainty aware agents plays a single episode storing the trajectory in the agents memory.
    """
    def __call__(self, agent, selection_strategy, memory):
        observation = agent.env.reset().detach()
        episode_reward = 0
        step_counter = 0
        terminate = False

        # episode_memory = Memory(['action', 'state', 'reward', 'new_state', 'terminal', 'uncertainty'],
        episode_memory = Memory(agent.train_loader.memory_cell_names,
                                gamma=memory.gamma)
        with eval_mode(agent):
            while not terminate:
                step_counter += 1
                agent.to(agent.device)
                action, certainty = selection_strategy(agent, observation.to(agent.device))
                new_observation, reward, terminate, _ = agent.env.step(action)

                episode_reward += torch.sum(reward).item() / agent.env.n_instances
                episode_memory.memorize((action,
                                         observation,
                                         torch.tensor(reward).float(),
                                         new_observation,
                                         terminate,
                                         certainty.detach()),
                                        ['action', 'state', 'reward', 'new_state', 'terminal', 'uncertainty'])
                observation = new_observation[~terminate.view(-1)]
                terminate = terminate.min().item()
        memory.memorize(episode_memory, episode_memory.memory_cell_names)
        agent.train_dict['rewards'] = agent.train_dict.get('rewards', []) + [episode_reward]

        if episode_reward > agent.train_dict.get('best_performance', -np.inf):
            agent.train_dict['best_performance'] = episode_reward

        return episode_reward


class DuelingDQNPlayer(RLPlayer):
    """
    This determines how a uncertainty aware agents plays a single episode storing the trajectory in the agents memory.
    """
    def __call__(self, agent, selection_strategy, memory):
        observation = agent.env.reset().detach()
        episode_reward = 0
        step_counter = 0
        terminate = False

        # episode_memory = Memory(['action', 'state', 'reward', 'new_state', 'terminal', 'uncertainty'],
        episode_memory = Memory(agent.train_loader.memory_cell_names,
                                gamma=memory.gamma)
        with eval_mode(agent):
            while not terminate:
                step_counter += 1
                agent.to(agent.device)
                action, val_certainty, advantage_certainty = selection_strategy(agent, observation.to(agent.device))
                new_observation, reward, terminate, _ = agent.env.step(action)

                episode_reward += torch.sum(reward).item() / agent.env.n_instances
                episode_memory.memorize((action,
                                         observation,
                                         torch.tensor(reward).float(),
                                         new_observation,
                                         terminate,
                                         val_certainty.detach(),
                                         advantage_certainty.detach()
                                         ),
                                        ['action', 'state', 'reward', 'new_state', 'terminal', 'value',
                                         'advantages'])
                observation = new_observation[~terminate.view(-1)]
                terminate = terminate.min().item()
        memory.memorize(episode_memory, episode_memory.memory_cell_names)
        agent.train_dict['rewards'] = agent.train_dict.get('rewards', []) + [episode_reward]

        if episode_reward > agent.train_dict.get('best_performance', -np.inf):
            agent.train_dict['best_performance'] = episode_reward

        return episode_reward