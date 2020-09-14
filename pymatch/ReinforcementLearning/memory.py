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
        self.reduce_buffer()

    def _memorize_values(self, values, cell_name: list):
        for val, cell in zip(values, cell_name):
            self.memory[cell] += [val]

    def _merge_memory(self, values, cell_name):
        for key in cell_name:
            self.memory[key] += values.memory[key]

    def reduce_buffer(self, reduce_to=None):
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
    def __init__(self, memory_refresh_rate):
        """
        Updates the memory of
        Args:
            memory_refresh_rate: fraction of oldest memories to be replaced when updated
        """
        if not 0. <= memory_refresh_rate <= 1.:
            raise ValueError(f'memory_refresh_rate was set to {memory_refresh_rate} but has to be in ]0., 1.]')
        self.memory_refresh_rate = memory_refresh_rate

    def __call__(self, agent):
        reduce_to = int(len(agent.memory) * (1 - self.memory_refresh_rate))
        agent.memory.reduce_buffer(reduce_to)
        self.fill_memory(agent)

    def fill_memory(self, agent):
        reward, games = 0, 0
        while len(agent.memory) < agent.memory.buffer_size:
            reward += agent.play_episode()
            games += 1
        agent.memory.reduce_buffer()
        agent.train_dict['avg_reward'] = agent.train_dict.get('avg_reward', []) + [reward / games]