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
                 replace=True,
                 ignore_col=[]):
        """
        Memory class for RL algorithm.

        Args:
            memory_cell_names:  key with which the memory can be accessed (necesseray when storing)
            n_samples:          samples that are drawn for each forward pass
            memory_cell_space:  dimensionality of each element stored in the memory (depricated @todo clean this up)
            memory_size:        max number of individual steps the memory can hold
            batch_size:         batch size used when drawing samples from the memory
            gamma:              discount factor used for REINFORCE
                                (@todo this is weird to have here since it is not used for other algorithms)
            replace:            Used when drawing samples from the memory. If set to true it replaces already drawn
                                memories. This can be used to create bootstrap samples of the memory, when it is used
                                by multiple agents at the same time.
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
        self.ignore_col = ignore_col
        self.nr_seen_states = 0

    def memorize(self, values, cell_name: list):
        """
        Memorizes a list of torch tensors to the memory cells provided by the cell_name list.

        Args:
            values (list)/(Memory): list of torch tensors to memorize
            cell_name (list): list of strings containing the memory cell names the tensors have to be added to

        """
        self.nr_seen_states += len(values)
        if len(np.unique([len(v) for v in values])) > 1:
            raise ValueError('Different length of values to memorize')
        if isinstance(values, Memory):  # create list of values from memory
            self.memory = self._merge_memory(values, cell_name, self.memory)
        else:
            self.memory = self._memorize_values(values, cell_name, self.memory)
        self.reduce_buffer()

    def _memorize_values(self, values, cell_name: list, memory):
        for val, cell in zip(values, cell_name):
            if type(val) != torch.Tensor:
                val = torch.tensor(val)
            if len(val.shape) == 0:
                val = val.unsqueeze(0)
            if len(val.shape) == 1:
                val = val.unsqueeze(0)
            if memory[cell] is None:
                memory[cell] = val
            else:
                memory[cell] = torch.cat([memory[cell], val])
        return memory

    def _merge_memory(self, values, cell_name, memory):
        """
        This merges two memories

        Args:
            values:     new memory that has to be merge into an already existing one (@todo rename variable)
            cell_name:  keys, that have to be merged
                        (@todo this is weird just check if the keys are the same and merge them then)
            memory:     memory the new memory has to be merged into

        Returns:
            Merged Memory
        """
        for key in cell_name:
            if memory[key] is None:
                memory[key] = values.memory[key]
            else:
                memory[key] = torch.cat([memory[key], values.memory[key]], 0)
        return memory

    def reduce_buffer(self, reduce_to=None):
        """
        Reduces the memory to its max capacity, throwing away the oldest memories.

        Args:
            reduce_to:  number of memories to reduce to. Setting this to 0 will reset the memory

        """
        reduce_to = reduce_to if reduce_to is not None else self.memory_size
        if reduce_to is not None:
            if reduce_to == 0:
                self.memory_reset()
            else:
                for key in self.memory:
                    self.memory[key] = self.memory[key][-reduce_to:]

    def get_size(self):
        """
        Gives back the size of the Memory. (@todo depricated and shouldn't be used anymore, use `len()` instead)

        Returns:
            Current of the memory

        """
        return self.__len__()

    def memory_reset(self):
        """
        Resets the memory to an empty one.
        The first element of each memory cell is always a default element.
        """
        self.memory = {key: None for key in self.memory_cell_names}

    def cumul_reward(self, cell_name='reward'):
        """
        Computes the cumulative reward of the memory.

        Args:
            cell_name (str):    name of the reward memory cell.

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
        if self.memory[list(self.memory.keys())[0]] is None:
            return 0
        return len(self.memory[list(self.memory.keys())[0]])

    def __getitem__(self, idx):
        result = []
        for key in self.memory:
            if key in self.ignore_col:
                continue
            result += [self.memory[key][idx]]
        return tuple(result)

    def sample_indices(self, n_samples):
        """
        Samples indices from the memory, either random shuffling the entire memory, or sampling with replacement, as
        determined in the constructor of the memory. This can be used for random subset necessary to create dataloader.

        Args:
            n_samples:  number of samples to draw

        Returns:
            numpy array of indices

        """
        if n_samples is None:
            idx = np.arange(self.__len__())
            np.random.shuffle(idx)
        else:
            n_samples = min(n_samples, self.__len__())
            idx = np.random.choice(range(self.__len__()), n_samples, replace=self.replace)
        return idx

    def sample_loader(self, n_samples=None, shuffle=True):
        """
        Samples a dataloader from the memory.

        Args:
            n_samples:  number of samples to draw (@todo we should create some default value as the full data set)
            shuffle:    determines if the memory should be shuffled

        Returns:

        """
        if shuffle:
            if n_samples is None:
                data_loader = torch.utils.data.DataLoader(self, batch_size=self.batch_size, shuffle=shuffle)
            else:
                data_loader = torch.utils.data.DataLoader(
                    self,
                    batch_size=self.batch_size,
                    sampler=SubsetRandomSampler(indices=self.sample_indices(n_samples=n_samples))
                )
        else:
            data_loader = torch.utils.data.DataLoader(self, batch_size=self.batch_size, shuffle=shuffle)
        return data_loader

    def __iter__(self):
        return iter(self.sample_loader(self.n_samples))

    def create_state_dict(self):
        """
        Creates a state dict of the memory to store.

        Returns:

        """
        return {'memory': self.memory}

    def restore_checkpoint(self, checkpoint):
        """
        Restores a previously stored checkpoint to the memory.

        Args:
            checkpoint: loaded checkpoint

        Returns:

        """
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
        if len(np.unique([len(v) for v in values])) > 1:
            raise ValueError('Different length of values to memorize')
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
    def __init__(self, memory_cell_names, temp=1., *args, **kwargs):
        kwargs['ignore_col'] = kwargs.get('ignore_col', ['uncertainty'])
        super().__init__(memory_cell_names, *args, **kwargs)
        # self.probability = None
        self.temp = temp

    def sample_indices(self, n_samples):
        # if self.probability is None:
        #     print('WARNNG: Probabilites for priority sampling are not set. This can be fine for the first iteration, '
        #           'but may not appear afterwards')
        if n_samples is None:
            n_samples = self.__len__()
        n_samples = min(n_samples, self.__len__())
        prob = self.compute_probs_from_certainty()
        idx = np.random.choice(range(self.__len__()), n_samples, p=prob.numpy(), replace=self.replace)
        return idx

    # def set_probability(self, probability):
    #     self.probability = probability

    def compute_probs_from_certainty(self):
        prob = self.memory['uncertainty'].max(-1)[0] + 1e-16 # this was added for numeric stability
        prob = (prob - prob.mean()) / prob.std()
        prob = torch.sigmoid(prob / self.temp)
        prob /= prob.sum()
        return prob

    # def __getitem__(self, idx):
    #     result = []
    #     for key in self.memory:
    #         if key == 'uncertainty':
    #             continue
    #         result += [self.memory[key][idx]]
    #     return tuple(result)