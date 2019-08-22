import _pickle
import gzip
import numpy as np


def pickle_dump(path, object):
    """
    dumping stuff using pickle with gzip compression
    """
    with gzip.open(path, 'wb') as output:
        _pickle.dump(object, output, protocol=-1)
    return


def pickle_load(path):
    """
    loading stuff dumped with pickle using gzip compression
    """
    with gzip.open(path, 'rb') as input:
        return _pickle.load(input)

def sliding_mean(values, window_size):
    values = np.array(values)
    return np.array([values[i - window_size : i].mean() for i in range(window_size, len(values))])
