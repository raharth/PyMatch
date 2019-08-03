import _pickle
import gzip


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