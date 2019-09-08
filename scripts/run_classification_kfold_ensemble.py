import torch
import torchvision as tv
from torch import nn
import torch.optim as optim

import matplotlib.pyplot as plt

from pytorch_lib.DeepLearning.Learner import ClassificationLearner
from models.test_Model import Model
from pytorch_lib.utils.KFold import KFold
from pytorch_lib.DeepLearning.Ensemble import Ensemble


def factory(kfold, device, lr, momentum, name, n_classes):
    model = Model(n_classes=n_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    crit = nn.CrossEntropyLoss()
    train_loader, val_loader = kfold.fold_loaders(fold=-1)
    learner = ClassificationLearner(model=model, optimizer=optimizer, crit=crit, train_loader=train_loader,
                                    val_loader=val_loader, name=name)
    return learner


# training the model
no_cuda = False
use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

dataset = tv.datasets.CIFAR10('./data/CIFAR10/', train=True, transform=tv.transforms.ToTensor(), download=False)

batch_size = 32
epochs = 5
lr = .01
momentum = .5
log_interval = 10
folds = 3
kfold = KFold(dataset=dataset, batch_size=batch_size, n_fold=folds)

params = {'kfold': kfold, 'device': device, 'lr': lr, 'momentum': momentum, 'name': 'kfold', 'n_classes': 10}

ensemble = Ensemble(trainer_factory=factory, n_model=folds, trainer_args=params)


ensemble.train(epochs=epochs, device=device)

for learner in ensemble.learners:
    plt.plot(learner.losses)
plt.show()

for learner in ensemble.learners:
    plt.plot(learner.train_accuracy)
plt.show()