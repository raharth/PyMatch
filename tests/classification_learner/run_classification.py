import torch
import torchvision as tv
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from pymatch.DeepLearning.hat import MaxProbabilityHat
from pymatch.DeepLearning.learner import ClassificationLearner
from pymatch.DeepLearning.pipeline import Pipeline
from pymatch.utils.DataHandler import DataHandler
import pymatch.DeepLearning.callback as cb


class Model(nn.Module):

    def __init__(self, n_classes):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 28 * 28, 500)
        self.bn3 = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(500, n_classes)
        self.ankered_layers = [self.fc1, self.fc2]

    def forward(self, X, train=True):
        if train:
            self.train()
        else:
            self.eval()

        x = self.conv1(X)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x.reshape(X.shape[0], -1)
        x = F.relu(x)
        x = self.fc1(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.fc2(x)
        out = F.softmax(x, dim=-1)
        return out

# training the model
no_cuda = False
use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

train_batch_size = 256
test_batch_size = 256
epochs = 2
lr = .01
momentum = .5
log_interval = 10


train_data = tv.datasets.CIFAR10('../data/CIFAR10/', train=True, transform=tv.transforms.ToTensor(), download=False)
test_data = tv.datasets.CIFAR10('../data/CIFAR10/', train=False, transform=tv.transforms.ToTensor(), download=False)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, shuffle=True)

crit = nn.CrossEntropyLoss()
crit_test = nn.CrossEntropyLoss(reduction='sum')

model = Model(len(train_data.classes)).to(device)

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

learner = ClassificationLearner(model=model,
                                optimizer=optimizer,
                                crit=crit,
                                train_loader=train_loader,
                                name='class_test',
                                load_checkpoint=False,
                                dump_path='tests/classification_learner',
                                callbacks=[cb.Checkpointer(),
                                           cb.EarlyStopping(test_loader),
                                           cb.EarlyTermination(patience=10),
                                           ])

learner.fit(epochs, device=device)

label_hat = MaxProbabilityHat()
pipeline = Pipeline(pipes=[learner, label_hat], pipe_args=[{'device': 'cuda'}, {}])

reporter = cb.Reporter(test_loader, folder_path='tests/classification_learner')
reporter(pipeline, train_data.classes)

plotter = cb.ConfusionMatrixPlotter(test_loader, img_path='tests/classification_learner')
plotter(pipeline, train_data.classes)


y_pred = pipeline.predict_dataloader(test_loader, device=device)
