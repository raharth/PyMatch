import random
import math
import numpy as np


class Sampler:

    def __int__(self):
        pass

    def sample(self):
        raise NotImplementedError


class IntervalSampler(Sampler):

    def __init__(self, lower_bound, upper_bound):
        super().__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.width = upper_bound - lower_bound

    def sample(self):
        return random.uniform(self.lower_bound, self.upper_bound)

    def shrink(self, center, shrinking):
        self.lower_bound = center - (self.width / shrinking) / 2
        self.upper_bound = center + (self.width / shrinking) / 2
        self.width = self.upper_bound - self.lower_bound


class LogSampler10(IntervalSampler):

    def __init__(self, lower_bound, upper_bound):
        super().__init__(math.log10(lower_bound), math.log10(upper_bound))

    def sample(self):
        return 10 ** super().sample()


class DiscreteSampler(Sampler):

    def __init__(self, values, probs=None):
        super().__init__()
        self.values = values
        self.probs = probs

    def sample(self):
        return self.values[np.random.choice(range(len(self.values)), p=self.probs)]

