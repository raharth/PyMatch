# well actually not a real test... at least some playing around with it

from pytorch_lib.utils.KFold import KFold
from torch.utils.data.dataset import Dataset
import torch


class MyCustomDataset(Dataset):

    def __init__(self, shape):
        size_ = 1
        for s in shape:
            size_ *= s
        self.X = torch.tensor(range(size_)).view(shape)
        self.y = self.X[:, :1]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.y)


class MyCustomDataset2(Dataset):

    def __init__(self, shape):
        size_ = 1
        for s in shape:
            size_ *= s
        self.X = torch.tensor([[i for _ in range(shape[1])] for i in range(shape[0])])
        self.y = self.X[:, :1]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.y)


if __name__ == '__main__':
    dataset = MyCustomDataset((10, 2))

    kfold = KFold(dataset=dataset, n_fold=2, batch_size=1)

    train_loader, test_loader = kfold.fold_loaders(0)

    # train_loader.dataset.y
    # test_loader.dataset.y

    for d, y in train_loader:
        print(d, y)

    print('train')
    for d, y in test_loader:
        print(d, y)


dataset = MyCustomDataset2((5, 2))
kfold = KFold(dataset=dataset, n_fold=5, batch_size=1)
for i in range(5):
    print('loader: {}'.format(i))
    train_loader, test_loader = kfold.fold_loaders(fold=-1)
    for d, y in test_loader:
        print(d, y)
    print('')



# shape = (10, 2)
# size = 1
# for s in shape:
#     size *= s
# X = torch.tensor(range(size)).view(shape)
# X
# y = X[:, :1]
# y