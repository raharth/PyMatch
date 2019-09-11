import torch
import torchvision as tv
from torch import nn
import torch.optim as optim

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

from models.test_Model import Model
from pytorch_lib.DeepLearning.Learner import ClassificationLearner
from pytorch_lib.utils.KFold import KFold
from pytorch_lib.DeepLearning.Ensemble import Ensemble
from pytorch_lib.utils.Functional import scale_confusion_matrix, plot_confusion_matrix


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
epochs = 20
lr = .01
momentum = .5
log_interval = 1
folds = 5
kfold = KFold(dataset=dataset, batch_size=batch_size, n_fold=folds)

params = {'kfold': kfold, 'device': device, 'lr': lr, 'momentum': momentum, 'name': 'kfold', 'n_classes': 10}

ensemble = Ensemble(trainer_factory=factory, n_model=folds, trainer_args=params)


# ensemble.train(epochs=epochs, device=device)
ensemble.load_checkpoint()

for learner in ensemble.learners:
    plt.plot(learner.losses)
plt.title('train loss')
plt.show()

for learner in ensemble.learners:
    plt.plot(learner.train_accuracy)
plt.title('train acc')
plt.show()

for learner in ensemble.learners:
    plt.plot(learner.val_losses)
plt.title('val loss')
plt.show()

for learner in ensemble.learners:
    plt.plot(learner.val_accuracy)
plt.title('val acc')
plt.show()

y_preds, y_trues = ensemble.run_validation(device=device)
cm = []
for y_pred, y_true in zip(y_preds, y_trues):
    cm += [scale_confusion_matrix(confusion_matrix(y_true, y_pred))]

cm = np.array(cm)
cm_mean = cm.mean(0)
cm_std = cm.std(0)

plot_confusion_matrix(cm[0], figsize=(10, 10))
plt.show()

plot_confusion_matrix(cm_mean, figsize=(10, 10))
plt.show()

plot_confusion_matrix(cm_std, figsize=(10, 10))
plt.show()