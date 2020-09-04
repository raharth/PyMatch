import torch


class KFold:

    def __init__(self, dataset, n_fold=10, batch_size=32, num_workers=0, pin_memory=False):
        self.fold = 0
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dataset = dataset
        self.n_fold = n_fold
        self.fold_size = len(self.dataset) // self.n_fold
        self.folded_size = self.n_fold * self.fold_size

        self.fold_idx = self.fold_split()

    def fold_split(self, random_seed=None):
        """
        Splitting the folds.

        Args:
            random_seed: Random seed for reproducibility

        Returns:
            tensor containing indices for folds, where dim=0 is the fold number

        """
        if random_seed is not None:
            torch.manual_seed(random_seed)
        fold_idx = torch.randperm(self.dataset.__len__())
        fold_idx = fold_idx[:self.folded_size].view(-1, self.fold_size)
        return fold_idx

    def fold_loaders(self, fold=-1):
        """
        Loading a specific fold as train and test data loader. If no fold number is provided it returns the next fold. It returns a randomly sampled subset of
        the original data set.

        Args:
            fold: fold number to return

        Returns:
            (train data loader, test data loader)

        """
        if fold == -1:
            fold = self.fold
        test_fold_idx = self.fold_idx[fold]
        train_fold_idx = self.fold_idx[[i for i in range(self.n_fold) if i != fold]].view(-1)
        train_loader = torch.utils.data.DataLoader(self.dataset,
                                                   batch_size=self.batch_size,  # args.batch_size,
                                                   num_workers=self.num_workers,  # args.loader_num_workers,
                                                   pin_memory=self.pin_memory,
                                                   sampler=torch.utils.data.SubsetRandomSampler(train_fold_idx))
        test_loader = torch.utils.data.DataLoader(self.dataset,
                                                  batch_size=self.batch_size,  # args.batch_size,
                                                  num_workers=self.num_workers,  # args.loader_num_workers,
                                                  pin_memory=self.pin_memory,
                                                  sampler=torch.utils.data.SubsetRandomSampler(test_fold_idx))
        self.fold = (self.fold + 1) % self.n_fold
        return train_loader, test_loader


