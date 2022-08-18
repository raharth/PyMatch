import torch
import torchvision as tv
from torch import nn
import torch.optim as optim

from pymatch.DeepLearning.hat import MaxProbabilityHat
from pymatch.DeepLearning.learner import ClassificationLearner
from pymatch.DeepLearning.callback import Reporter, ConfusionMatrixPlotter
from pymatch.DeepLearning.pipeline import Pipeline
from pymatch.utils.DataHandler import DataHandler
from models.test_Model import Model

# training the model
no_cuda = False
use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

train_batch_size = 64
test_batch_size = 1000
epochs = 10
lr = .01
momentum = .5
log_interval = 10


train_data = tv.datasets.CIFAR10('./data/CIFAR10/', train=True, transform=tv.transforms.ToTensor(), download=False)
test_data = tv.datasets.CIFAR10('data/CIFAR10/', train=False, transform=tv.transforms.ToTensor(), download=False)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, shuffle=True)

crit = nn.CrossEntropyLoss()

model = Model(len(train_data.classes)).to(device)

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

learner = ClassificationLearner(model=model, optimizer=optimizer, crit=crit, train_loader=train_loader,
                                val_loader=test_loader, name='class_test', load_checkpoint=True)

learner.fit(epochs, device=device)

label_hat = MaxProbabilityHat()
pipeline = Pipeline(pipes=[learner, label_hat], pipe_args=[{'device': 'cuda'}, {}])

reporter = Reporter(test_loader)
reporter.__call__(pipeline, train_data.classes)

plotter = ConfusionMatrixPlotter(test_loader)
plotter.__call__(pipeline, train_data.classes)


y_pred = pipeline.predict_dataloader(test_loader, device=device)
