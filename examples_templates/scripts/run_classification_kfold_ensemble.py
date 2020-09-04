import torch
import torchvision as tv
from torch import nn
import torch.optim as optim

from models.test_Model import Model
from pymatch.DeepLearning.callback import ConfusionMatrixPlotter, Reporter
from pymatch.DeepLearning.learner import ClassificationLearner
from pymatch.utils.KFold import KFold
from pymatch.DeepLearning.ensemble import Ensemble
from pymatch.utils.Functional import read_setting


def factory(kfold, device, lr, momentum, name, n_classes):
    model = Model(n_classes=n_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    weights = torch.ones(10) # .to(device)
    crit = nn.CrossEntropyLoss(weight=weights).to(device)
    train_loader, val_loader = kfold.fold_loaders(fold=-1)
    learner = ClassificationLearner(model=model, optimizer=optimizer, crit=crit, train_loader=train_loader,
                                    val_loader=val_loader, name=name)
    return learner


# training the model
no_cuda = False
use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

epochs = 10

dataset = tv.datasets.CIFAR10('./data/CIFAR10/', train=True, transform=tv.transforms.ToTensor(), download=False)

settings = read_setting('setting.json')

kfold = KFold(dataset=dataset, batch_size=settings['batch_size'], n_fold=settings['folds'])

params = {'kfold': kfold, 'device': device, 'lr': settings['lr'], 'momentum': settings['momentum'],
          'name': 'kfold', 'n_classes': len(dataset.classes)}

ensemble = Ensemble(trainer_factory=factory, n_model=settings['folds'], trainer_args=params)


ensemble.fit(epochs=epochs, device=device)
ensemble.load_checkpoint()
classes = dataset.classes

cm_plotter = ConfusionMatrixPlotter(data_loader=kfold.fold_loaders(0))
cm_plotter.__call__(model=ensemble, classes=classes, device=device)

reporter = Reporter(data_loader=kfold.fold_loaders(0))
reporter.__call__(model=ensemble, classes=classes)
