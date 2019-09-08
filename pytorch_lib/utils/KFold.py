import torch


class KFold:

    def __init__(self, dataset, n_fold=10, batch_size=32, num_workers=0):
        self.fold = 0
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = dataset
        self.n_fold = n_fold
        self.fold_size = len(self.dataset) // self.n_fold
        self.folded_size = self.n_fold * self.fold_size

        self.fold_idx = self.fold_split()

    def fold_split(self):
        fold_idx = torch.randperm(self.dataset.__len__())
        fold_idx = fold_idx[:self.folded_size].view(-1, self.fold_size)
        return fold_idx

    def fold_loaders(self, fold=-1):
        if fold == -1:
            fold = self.fold
        test_fold_idx = self.fold_idx[fold]
        train_fold_idx = self.fold_idx[[i for i in range(self.n_fold) if i != fold]].view(-1)
        train_loader = torch.utils.data.DataLoader(self.dataset,
                                                   batch_size=self.batch_size,  # args.batch_size,
                                                   num_workers=self.num_workers,  # args.loader_num_workers,
                                                   sampler=torch.utils.data.SubsetRandomSampler(train_fold_idx))
        test_loader = torch.utils.data.DataLoader(self.dataset,
                                                  batch_size=self.batch_size,  # args.batch_size,
                                                  num_workers=self.num_workers,  # args.loader_num_workers,
                                                  sampler=torch.utils.data.SubsetRandomSampler(test_fold_idx))
        self.fold = (self.fold + 1) % self.n_fold
        return train_loader, test_loader







#
# # dataset = tv.datasets.CIFAR10('./data/CIFAR10/', train=True, transform=tv.transforms.ToTensor(), download=True)
# dataset = torch.tensor(range(40)).view(-1, 2)
# n_fold = 4
# fold_size = len(dataset) // n_fold
# folded_size = n_fold * fold_size
# fold = 1
#
# fold_idx = torch.randperm(folded_size)  # permutation
# fold_idx = torch.tensor(range(folded_size)) # unpermuted
# fold_idx = fold_idx[:folded_size].view(-1, fold_size)
#
# test_fold_idx = fold_idx[fold]
# train_fold_idx = fold_idx[[i for i in range(n_fold) if i != fold]].view(-1)

