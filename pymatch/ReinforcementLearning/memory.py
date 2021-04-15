import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd


class Memory(Dataset):

    def __init__(self,
                 memory_cell_names,
                 n_samples=None,
                 memory_cell_space=None,
                 memory_size=None,
                 batch_size=64,
                 gamma=.0,
                 replace=True):
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
        self.replace = replace
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

    def _merge_memory(self, values, cell_name, memory):
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
        if n_samples is None:
            idx = np.arange(self.__len__())
            np.random.shuffle(idx)
        else:
            n_samples = min(n_samples, self.__len__())
            idx = np.random.choice(range(self.__len__()), n_samples, replace=self.replace)
        return idx

    def sample_loader(self, n_samples, shuffle=True):
        if shuffle:
            data_loader = torch.utils.data.DataLoader(
                self,
                batch_size=self.batch_size,
                sampler=SubsetRandomSampler(indices=self.sample_indices(n_samples=n_samples))
            )
        else:
            data_loader = torch.utils.data.DataLoader(self, batch_size=self.batch_size)
        return data_loader

    def __iter__(self):
        return iter(self.sample_loader(self.n_samples))

    def create_state_dict(self):
        return {'memory': self.memory}

    def restore_checkpoint(self, checkpoint):
        self.memory = checkpoint['memory']


class StateTrackingMemory(Memory):

    def __init__(self,
                 memory_cell_names,
                 n_samples=None,
                 memory_cell_space=None,
                 memory_size=None,
                 batch_size=64,
                 gamma=.0,
                 root='.',
                 detach_tensors=[]):
        super().__init__(memory_cell_names=memory_cell_names,
                         n_samples=n_samples,
                         memory_cell_space=memory_cell_space,
                         memory_size=memory_size,
                         batch_size=batch_size,
                         gamma=gamma)
        self.eternal_memory = {k: [] for k in memory_cell_names + ['iteration']}
        self.root = root
        self.detach_tensors = detach_tensors
        self.iteration = 0

    def memorize(self, values, cell_name: list):
        """
        Memorizes a list of torch tensors to the memory cells provided by the cell_name list.

        Args:
            values (list)/(Memory): list of torch tensors to memorize
            cell_name (list): list of strings containing the memory cell names the tensors have to be added to

        """
        super(StateTrackingMemory, self).memorize(values, cell_name)

        if isinstance(values, Memory):  # create list of values from memory
            for detach in self.detach_tensors:
                values.memory[detach] = values.memory[detach].detach()
            values.memory['iteration'] = [self.iteration] * len(values)
            self.eternal_memory = self._merge_memory(values, cell_name + ['iteration'], self.eternal_memory)
        else:
            for detach in self.detach_tensors:
                idx = np.argwhere(np.array(cell_name) == detach).item()
                values[idx] = values[idx].detach()
            self.eternal_memory = self._memorize_values(values + [self.iteration], cell_name + ['iteration'], self.eternal_memory)

        self.iteration += 1


class PriorityMemory(Memory):
    def __init__(self, memory_cell_names, *args, **kwargs):
        super().__init__(memory_cell_names, *args, **kwargs)
        self.probability = None

    def sample_indices(self, n_samples):
        # if self.probability is None:
        #     print('WARNNG: Probabilites for priority sampling are not set. This can be fine for the first iteration, '
        #           'but may not appear afterwards')
        if n_samples is None:
            n_samples = self.__len__()
        n_samples = min(n_samples, self.__len__())
        prob = self.memory['certainty']
        idx = np.random.choice(range(self.__len__()), n_samples, p=prob, replace=self.replace)
        return idx

    def set_certainty(self, probability):
        self.probability = probability
