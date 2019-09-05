import torch


class DataHandler:

    @staticmethod
    def split(dataset, test_frac):

        test_size = int(test_frac * len(dataset))
        train_size = len(dataset) - test_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        return train_dataset, test_dataset

    @staticmethod
    def even_split(dataset, test_frac):
        # @todo
        pass