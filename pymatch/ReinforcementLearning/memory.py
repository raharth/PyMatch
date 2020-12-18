import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler


class Memory(Dataset):

    def __init__(self,
                 memory_cell_names,
                 n_samples=None,
                 memory_cell_space=None,
                 memory_size=None,
                 batch_size=64,
                 gamma=.0):
        """
        Memory class for RL algorithm.

        Args:
            memory_size (int): max buffer size

        """
        if memory_cell_space is None:
            memory_cell_space = [1 for _ in range(len(memory_cell_names))]
        self.memory_cell_space = memory_cell_space
        self.memory_cell_names = memory_cell_names
        self.memory_size = memory_size
        if gamma >= 1.:
            raise ValueError('gamma is larger then 1 which will lead to exponential growth in rewards')
        self.gamma = gamma
        self.memory = {}
        self.memory_reset()
        self.batch_size = batch_size
        self.n_samples = n_samples

    def memorize(self, values, cell_name: list):
        """
        Memorizes a list of torch tensors to the memory cells provided by the cell_name list.

        Args:
            values (list)/(Memory): list of torch tensors to memorize
            cell_name (list): list of strings containing the memory cell names the tensors have to be added to

        """
        if isinstance(values, Memory):  # create list of values from memory
            self.memory = self._merge_memory(values, cell_name, self.memory)
        else:
            self.memory = self._memorize_values(values, cell_name, self.memory)
        self.reduce_buffer()

    def _memorize_values(self, values, cell_name: list, memory):
        for val, cell in zip(values, cell_name):
            memory[cell] += [val]
        return memory

    def _merge_memory(self, values, cell_name):
        memory = dict(self.memory)
        for key in cell_name:
            memory[key] += values.memory[key]
        return memory

    def reduce_buffer(self, reduce_to=None):
        """
        Reduces the memory to its max capacity.

        """
        reduce_to = reduce_to if reduce_to is not None else self.memory_size
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

    def cumul_reward(self, cell_name='reward'):
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
            R = R * self.gamma + r.item()
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

    def sample_loader(self, n_samples):
        return torch.utils.data.DataLoader(
            self,
            batch_size=self.batch_size,
            sampler=SubsetRandomSampler(indices=self.sample_indices(n_samples=n_samples))
        )

    def __iter__(self):
        return iter(self.sample_loader(self.n_samples))

    def create_state_dict(self):
        return {'memory': self.memory}

    def restore_checkpoint(self, checkpoint):
        self.memory = checkpoint['memory']


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
        reduce_to = int(len(agent.train_loader) * (1 - self.memory_refresh_rate))
        agent.train_loader.reduce_buffer(reduce_to)
        self.fill_memory(agent)

    def fill_memory(self, agent):
        reward, games = 0, 0
        while len(agent.train_loader) < agent.train_loader.memory_size:
            reward += agent.play_episode()
            games += 1
        agent.train_loader.reduce_buffer()
        agent.train_dict['avg_reward'] = agent.train_dict.get('avg_reward', []) + [reward / games]


class StateTrackingMemory(Memory):

    def __init__(self,
                 memory_cell_names,
                 n_samples=None,
                 memory_cell_space=None,
                 memory_size=None,
                 batch_size=64,
                 gamma=.0):
        super().__init__(memory_cell_names=memory_cell_names,
                         n_samples=n_samples,
                         memory_cell_space=memory_cell_space,
                         memory_size=memory_size,
                         batch_size=batch_size,
                         gamma=gamma)
        self.eternal_memory = {k: [] for k in memory_cell_names + ['update']}

    def memorize(self, values, cell_name: list):
        """
        Memorizes a list of torch tensors to the memory cells provided by the cell_name list.

        Args:
            values (list)/(Memory): list of torch tensors to memorize
            cell_name (list): list of strings containing the memory cell names the tensors have to be added to

        """
        if isinstance(values, Memory):  # create list of values from memory
            self.eternal_memory = self._merge_memory(values, cell_name, self.eternal_memory)
        else:
            self.eternal_memory = self._memorize_values(values, cell_name, self.eternal_memory)
        super(StateTrackingMemory, self).memorize(values, cell_name)

    def create_state_dict(self):
        state_dict = super().create_state_dict()
        state_dict['eternal_memory'] = self.eternal_memory
        return state_dict

    def restore_checkpoint(self, checkpoint):
        super().restore_checkpoint(checkpoint)
        self.eternal_memory = checkpoint['eternal_memory']