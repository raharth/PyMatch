import torch
import numpy as np


class Memory:

    def __init__(self, memory_cell_names, memory_cell_space=None, buffer_size=None):
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
        self.memory = {}
        for key in zip(self.memory_cell_names):
            self.memory[key] = []

    def _reduce_buffer(self):
        """
        Reduces the memory to its max capacity.

        """
        if self.buffer_size is not None:
            for key in self.memory:
                self.memory[key] = self.memory[key][-self.buffer_size:]

    def sample(self, n=None, replace=True):
        """
        Samples memories. It returns a list of torch.tensors where the order of the list elements is provided by the order of the memory cells.

        Args:
            n (int): number of memories sampled. If no value is provided the entire memory is returned but shuffled
            replace (bool): sample with or without replacement

        Returns:
            a list of all sampled memory elements as torch.tensors
        """
        curr_size = self.get_size()
        if n is None:
            n = curr_size   # first element is a default element
            replace = False
        mask = np.random.choice(range(curr_size), n, replace=replace)
        result = []

        memory = {}
        for key in self.memory:
            memory[key] = torch.stack(self.memory[key])

        for key in memory:
            result += [memory[key][mask]]
        return result

    def get_size(self):
        return len(self.memory[list(self.memory.keys())[0]])

    def memory_reset(self):
        """
        Resets the memory to an empty one.
        The first element of each memory cell is always a default element.
        """
        self.memory = {}
        for key in self.memory_cell_names:
            self.memory[key] = []

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


class PGMemory(Memory):

    def __init__(self, memory_cell_names, memory_cell_space=None, buffer_size=None):
        super(PGMemory, self).__init__(memory_cell_names, memory_cell_space=memory_cell_space, buffer_size=buffer_size)

    def memory_reset(self):
        """
        Resets the memory to an empty one.
        The first element of each memory cell is always a default element.
        """
        self.memory = {}
        for key, space in zip(self.memory_cell_names, self. memory_cell_space):
            self.memory[key] = torch.zeros(space)

    def cumul_reward(self, cell_name='reward', gamma=.95):
        """
        Computes the cumulative reward of the memory.

        Args:
            cell_name (str): name of the reward memory cell.

        Returns:
            None

        """
        reward = self.memory[cell_name]
        Reward = []
        R = 0
        for r in reward.flip(0):
            R = R * gamma + r.item()
            Reward.insert(0, R)
        self.memory[cell_name] = torch.tensor(Reward)


class QMemory(Memory):

    def __init__(self, memory_cell_names, memory_cell_space=None, buffer_size=None):
        super(QMemory, self).__init__(memory_cell_names, memory_cell_space=memory_cell_space, buffer_size=buffer_size)

    def memory_reset(self):
        """
        Resets the memory to an empty one.
        The first element of each memory cell is always a default element.
        """
        self.memory = {}
        for key, space in zip(self.memory_cell_names, self. memory_cell_space):
            self.memory[key] = torch.zeros(1, space)