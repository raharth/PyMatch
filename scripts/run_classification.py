import torch
import torchvision as tv
from torch import nn
import torch.optim as optim

from pytorch_lib import ClassificationLearner
from models.test_Model import Model

# training the model
no_cuda = False
use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

train_batch_size = 256
test_batch_size = 1000
epochs = 30
lr = .01
momentum = .5
log_interval = 10


train_data = tv.datasets.CIFAR10('data/CIFAR10/', train=True, transform=tv.transforms.ToTensor(), download=True)
test_data = tv.datasets.CIFAR10('data/CIFAR10/', train=False, transform=tv.transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, shuffle=True)

crit = nn.CrossEntropyLoss()
crit_test = nn.CrossEntropyLoss(reduction='sum')

model = Model(10).to(device)

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
loss_train = []
loss_test = []
accuracy = []


trainer = ClassificationLearner(model=model, optimizer=optimizer, crit=crit, train_loader=train_loader,
                                val_loader=test_loader, name='class_test', load_checkpoint=False)

trainer.train(epochs, device=device)