import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import numpy as np

from pymatch.ReinforcementLearning.torch_gym import TorchGym


class Memory(Dataset):

    def __init__(self,
                 memory_cell_names,
                 memory_cell_space=None,
                 buffer_size=None,
                 batch_size=64,
                 gamma=1.0):
        """
        Memory class for RL algorithm.

        Args:
            buffer_size (int): max buffer size

        """
        if memory_cell_space is None:
            memory_cell_space = [1 for _ in range(len(memory_cell_names))]
        self.memory_cell_space = memory_cell_space
        self.memory_cell_names = memory_cell_names
        self.buffer_size = buffer_size
        if gamma < 1.:
            raise ValueError('gamma is larger then 1 which will lead to exponential growth in rewards')
        self.gamma = gamma
        self.memory = {}
        self.memory_reset()

    def memorize(self, values, cell_name: list):
        """
        Memorizes a list of torch tensors to the memory cells provided by the cell_name list.

        Args:
            values (list)/(Memory): list of torch tensors to memorize
            cell_name (list): list of strings containing the memory cell names the tensors have to be added to

        """
        if isinstance(values, Memory):  # create list of values from memory
            self._merge_memory(values, cell_name)
        else:
            self._memorize_values(values, cell_name)
        self._reduce_buffer()

    def _memorize_values(self, values, cell_name: list):
        for val, cell in zip(values, cell_name):
            self.memory[cell] += [val]

    def _merge_memory(self, values, cell_name):
        for key in cell_name:
            self.memory[key] += values.memory[key]

    def memory_reset(self):
        """
        Resets the memory to an empty one.
        The first element of each memory cell is always a default element.
        """
        self.memory = {key: [] for key in self.memory_cell_names}

    def _reduce_buffer(self, reduce_to=None):
        """
        Reduces the memory to its max capacity.

        """
        reduce_to = reduce_to if reduce_to is not None else self.buffer_size
        if reduce_to is not None:
            if reduce_to == 0:
                self.memory_reset()
            else:
                for key in self.memory:
                    self.memory[key] = self.memory[key][-reduce_to:]

    # def sample(self, n=None, replace=True):
    #     """
    #     @todo this should be done differently using the dataloader class instead
    #     Samples memories. It returns a list of torch.tensors where the order of the list elements is provided by the order of the memory cells.
    #
    #     Args:
    #         n (int): number of memories sampled. If no value is provided the entire memory is returned but shuffled
    #         replace (bool): sample with or without replacement
    #
    #     Returns:
    #         a list of all sampled memory elements as torch.tensors
    #     """
    #     curr_size = self.get_size()
    #     if n is None:
    #         n = curr_size   # first element is a default element
    #         replace = False
    #     mask = np.random.choice(range(curr_size), n, replace=replace)
    #     result = []
    #
    #     memory = {}
    #     for key in self.memory:
    #         memory[key] = torch.stack(self.memory[key])
    #
    #     for key in memory:
    #         result += [memory[key][mask]]
    #     return result

    def get_size(self):
        return self.__len__()

    def memory_reset(self):
        """
        Resets the memory to an empty one.
        The first element of each memory cell is always a default element.
        """
        self.memory = {key: [] for key in self.memory_cell_names}

    def cumul_reward(self, cell_name='reward', gamma=.95):
        """
        Computes the cumulative reward of the memory.

        Args:
            cell_name (str): name of the reward memory cell.

        Returns:
            None

        """
        reward = torch.stack(self.memory[cell_name])
        Reward = []
        R = 0
        for r in reward.flip(0):
            R = R * gamma + r.item()
            Reward.insert(0, torch.tensor(R))
        self.memory[cell_name] = Reward

    def __len__(self):
        return len(self.memory[list(self.memory.keys())[0]])

    def __getitem__(self, idx):
        result = []
        for key in self.memory:
            result += [self.memory[key][idx]]
        return tuple(result)

    def sample_indices(self, n_samples):
        return np.random.choice(range(self.__len__()), n_samples)


class MemoryUpdater:
    def __init__(self, memory_refresh_rate, update_frequ=1.):
        super().__init__()
        if not 0. <= memory_refresh_rate <= 1.:
            raise ValueError(f'memory_refresh_rate was set to {memory_refresh_rate} but has to be in ]0., 1.]')
        self.memory_refresh_rate = memory_refresh_rate
        self.update_frequ = update_frequ

    def __call__(self, agent):
        if agent.train_dict['epochs_run'] % self.update_frequ == 0:
            reduce_to = int(len(agent.memory) * (1 - self.memory_refresh_rate))
            agent.memory._reduce_buffer(reduce_to)
            self.fill_memory(agent)

    def fill_memory(self, agent):
        while len(agent.memory) < agent.memory.buffer_size:
            game = self.play_episode(agent=agent)
            # agent.memory.memorize(game, ['log_prob', 'reward'])
        agent.memory._reduce_buffer()

    def play_episode(self, agent):
        observation = agent.env.reset().detach()
        episode_reward = 0
        step_counter = 0
        terminate = False
        episode_memory = Memory(['log_prob', 'reward'])

        while not terminate:
            step_counter += 1
            action, log_prob = agent.chose_action(observation) #.to(agent.device))
            new_observation, reward, done, _ = agent.env.step(action)

            episode_reward += reward
            episode_memory.memorize((log_prob, torch.tensor(reward)), ['log_prob', 'reward'])
            observation = new_observation
            terminate = done or (agent.env.max_episode_length is not None
                                 and step_counter >= agent.env.max_episode_length)

            # agent.env.render()
            if done:
                break

        episode_memory.cumul_reward(gamma=agent.gamma)
        agent.memory.memorize(episode_memory, episode_memory.memory_cell_names)
        agent.train_dict['rewards'] = agent.train_dict.get('rewards', []) + [episode_reward]
        return episode_reward


d = MemoryUpdater(memory_refresh_rate=.1)


# class PGMemory(Memory):
#
#     def __init__(self, memory_cell_names, memory_cell_space=None, buffer_size=None):
#         super(PGMemory, self).__init__(memory_cell_names, memory_cell_space=memory_cell_space,
#         buffer_size=buffer_size)
#
#     def memory_reset(self):
#         """
#         Resets the memory to an empty one.
#         The first element of each memory cell is always a default element.
#         """
#         self.memory = {}
#         for key, space in zip(self.memory_cell_names, self. memory_cell_space):
#             self.memory[key] = torch.zeros(space)
#
#     def cumul_reward(self, cell_name='reward', gamma=.95):
#         """
#         Computes the cumulative reward of the memory.
#
#         Args:
#             cell_name (str): name of the reward memory cell.
#
#         Returns:
#             None
#
#         """
#         reward = self.memory[cell_name]
#         Reward = []
#         R = 0
#         for r in reward.flip(0):
#             R = R * gamma + r.item()
#             Reward.insert(0, R)
#         self.memory[cell_name] = torch.tensor(Reward)


# class QMemory(Memory):
#
#     def __init__(self, memory_cell_names, memory_cell_space=None, buffer_size=None):
#         super(QMemory, self).__init__(memory_cell_names, memory_cell_space=memory_cell_space, buffer_size=buffer_size)
#
#     def memory_reset(self):
#         """
#         Resets the memory to an empty one.
#         The first element of each memory cell is always a default element.
#         """
#         self.memory = {}
#         for key, space in zip(self.memory_cell_names, self. memory_cell_space):
#             self.memory[key] = torch.zeros(1, space)
