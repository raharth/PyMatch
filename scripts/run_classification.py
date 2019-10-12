import torch
import torchvision as tv
from torch import nn
import torch.optim as optim

from pytorch_lib.DeepLearning.Hat import LabelHat, DefaultClassHat
from pytorch_lib.DeepLearning.HatCord import HatCord
from pytorch_lib.DeepLearning.Learner import ClassificationLearner
from pytorch_lib.DeepLearning.Callback import Reporter, ConfusionMatrixPlotter
from pytorch_lib.DeepLearning.Pipeline import Pipeline
from pytorch_lib.utils.DataHandler import DataHandler
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
crit_test = nn.CrossEntropyLoss(reduction='sum')

model = Model(len(train_data.classes)).to(device)

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

learner = ClassificationLearner(model=model, optimizer=optimizer, crit=crit, train_loader=train_loader,
                                val_loader=test_loader, name='class_test', load_checkpoint=True)

# learner.train(epochs, device=device)

label_hat = LabelHat()
label_learner = HatCord(learner, [label_hat])

reporter = Reporter(test_loader)
reporter.callback(label_learner, train_data.classes)

plotter = ConfusionMatrixPlotter(test_loader)
plotter.callback(label_learner, train_data.classes)

pipeline = Pipeline(pipes=[learner, label_hat], pipe_args=[{'device': 'cuda'}, {}])

y_pred = pipeline.predict_dataloader(test_loader, device=device)

# y_pred = DataHandler.predict_data_loader(learner, test_loader, device='cuda')

# label_hat = LabelHat()
# y_labeled = label_hat.cover(y_pred)

# default_hat = DefaultClassHat()
# y_default = default_hat.cover(y_pred)
