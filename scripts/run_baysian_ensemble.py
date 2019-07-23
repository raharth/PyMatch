import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as tv

from DeepLearning.Ensemble import BaysianEnsemble
from DeepLearning.Trainer import ClassificationTrainer
from DeepLearning.Loss import L2Loss, AnkerLoss
from models.test_Model import Model

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def trainer_factory(train_loader, val_loader, device, lr, momentum, name, n_classes, C=.1):
    """ Factory for trainer object"""
    model = Model(n_classes=n_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    crit = AnkerLoss(crit=nn.CrossEntropyLoss(), model=model, C=C)
    trainer = ClassificationTrainer(model=model, optimizer=optimizer, crit=crit, train_loader=train_loader,
                                    val_loader=val_loader, name=name)
    return trainer


# training the model
no_cuda = False
use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

train_batch_size = 64
test_batch_size = 1000
epochs = 20
lr = .01
momentum = .5
log_interval = 2
n_model = 20
checkpoint_int = 10
validation_int = 1
lambda_ = .0001

train_data = tv.datasets.CIFAR10('data/CIFAR10/', train=True, transform=tv.transforms.ToTensor(), download=True)
test_data = tv.datasets.CIFAR10('data/CIFAR10/', train=False, transform=tv.transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, shuffle=False)

trainer_args = {'train_loader': train_loader,
                'val_loader': test_loader,
                'device': device,
                'lr': lr,
                'momentum': momentum,
                'n_classes': 10,
                'C': lambda_
                }

ensemble = BaysianEnsemble(trainer_factory=trainer_factory, trainer_args=trainer_args, n_model=n_model)
ensemble.train(epochs=epochs, device=device, checkpoint_int=checkpoint_int, validation_int=validation_int, verbose=1,
               restore_early_stopping=True)

# ensemble.load_checkpoint(path='./tmp/early_stopping', tag='early_stopping')

y_pred_list = []
y_pred_s_list = []
correct_pred = []
for data, y in tqdm(test_loader):
    y_pred, y_pred_s = ensemble.predict(data, device=device)
    y_pred_list += [y_pred]
    y_pred_s_list += [y_pred_s]
    correct_pred += [y == y_pred]
y_pred_list = torch.cat(y_pred_list )
y_pred_s_list = torch.cat(y_pred_s_list)
correct_pred = torch.cat(correct_pred)

print('accuracy: {}'.format(correct_pred.float().mean()))
print('correct - mean: {}, std: {}'.format(y_pred_s_list[correct_pred].mean(), y_pred_s_list[correct_pred].std()))
print('incorrect - mean: {}, std: {}'.format(y_pred_s_list[~correct_pred].mean(), y_pred_s_list[~correct_pred].std()))


# np.set_printoptions(suppress=True)
# idx = 7
# print('pred probas: {}'.format(y_pred_m[idx].numpy()))
# print('certainty: {}'.format(y_pred_s[idx].numpy()))
# print('correct class: {}'.format(train_data.targets[idx]))
# correct = y_pred == y
# accuracy = correct.float().mean()
# y_pred_m[correct]
# y_pred_val[correct]

# for y_t, y_p, y_s in zip(y, y_pred, y_pred_s):
#     print('correct: {}, y: {:.2f}, y_pred: {:.2f}, y_s: {:.2f}'.format(y_t == y_p.long(), y_t.item(), y_p.item(), y_s.item()))
