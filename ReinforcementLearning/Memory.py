import torch
import numpy as np


class Memory:

    def __init__(self, memory_cells, buffer_size=None):
        """
        Memory class for RL algorithm.

        Args:
            buffer_size (int): max buffer size

        """
        self.memory_cells = memory_cells
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
            values = [values.memory[key] for key in values.memory_cells]

        for val, cell in zip(values, cell_name):
            if len(val.shape) == 0:
                val = val.unsqueeze(0)
            self.memory[cell] = torch.cat((self.memory[cell], val))
        self._reduce_buffer()

    def memory_reset(self):
        """
        Resets the memory to an empty one.
        The first element of each memory cell is always a default element.
        """
        self.memory = {}
        for key in self.memory_cells:
            self.memory[key] = torch.tensor([0.])

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
            n = curr_size - 1   # first element is a default element
            replace = False
        mask = np.random.choice(range(1, curr_size), n, replace=replace)
        result = []
        for key in self.memory:
            result += [self.memory[key][mask]]
        return result

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

    def get_size(self):
        return len(self.memory[list(self.memory.keys())[0]])

