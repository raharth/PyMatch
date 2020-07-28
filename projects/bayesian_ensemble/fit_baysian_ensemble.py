import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as tv

from pytorch_lib.DeepLearning.ensemble import BaysianEnsemble
from pytorch_lib.DeepLearning.hat import MaxProbabilityHat, EnsembleHatStd
from pytorch_lib.DeepLearning.learner import ClassificationLearner
from pytorch_lib.DeepLearning.loss import AnkerLossClassification
from pytorch_lib.DeepLearning.callback import EnsembleLearningCurvePlotter
from models.test_Model import Model


# def predict_single_models(ensemble, data_loader):
#     for i, trainer in enumerate(ensemble.trainers):
#         y_pred_list = []
#         correct_pred = []
#
#         for data, y in tqdm(data_loader):
#             y_pred = trainer.forward(data)
#             y_pred_list += [y_pred]
#             correct_pred += [y == y_pred.to('cpu')]
#
#         y_pred_list = torch.cat(y_pred_list)
#         correct_pred = torch.cat(correct_pred)
#
#         print('{}: accuracy: {}'.format(i, correct_pred.float().mean()))


# def get_confidence(y_mean, y_std):
#     y_prob, y_pred = torch.max(y_mean, 1)
#     y_confidence = []
#     for y_p, y_s in zip(y_pred, y_std):
#         y_confidence += [y_s[y_p]]
#     return y_pred, y_prob, torch.stack(y_confidence)


def trainer_factory(train_loader, val_loader, device, lr, momentum, name, n_classes, C=.1, H=1., path='./tmp'):
    """ Factory for trainer object"""
    model = Model(n_classes=n_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    crit = AnkerLossClassification(crit=nn.CrossEntropyLoss(), model=model, C=C, device=device, H=H)
    trainer = ClassificationLearner(model=model,
                                    optimizer=optimizer,
                                    crit=crit,
                                    train_loader=train_loader,
                                    val_loader=val_loader,
                                    name=name,
                                    dump_path=path)
    return trainer

PROJECT_ROOT = 'projects/bayesian_ensemble'

# training the model
no_cuda = False
use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


train_batch_size = 64
test_batch_size = 1000
epochs = 30
lr = .01
momentum = .5
log_interval = 2
n_model = 5
checkpoint_int = 10
validation_int = 1
lambda_ = .5
n_classes = 10
H = 1.

train_data = tv.datasets.CIFAR10('../data/CIFAR10/', train=True, transform=tv.transforms.ToTensor(), download=True)
test_data = tv.datasets.CIFAR10('../data/CIFAR10/', train=False, transform=tv.transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, shuffle=False)

trainer_args = {'train_loader': train_loader,
                'val_loader': test_loader,
                'device': device,
                'lr': lr,
                'momentum': momentum,
                'n_classes': n_classes,
                'C': lambda_,
                'H': H,
                "path": f'{PROJECT_ROOT}/tmp'
                }

ensemble = BaysianEnsemble(trainer_factory=trainer_factory,
                           trainer_args=trainer_args,
                           n_model=n_model,
                           callbacks=[EnsembleLearningCurvePlotter(target_folder_path=f'{PROJECT_ROOT}/tmp')])
ensemble.fit(epochs=epochs,
             device=device,
             checkpoint_int=checkpoint_int,
             validation_int=validation_int,
             verbose=1,
             restore_early_stopping=False,
             callback_iter=1)

ensemble.load_checkpoint(path='./projects/bayesian_ensemble/pretrained/early_stopping', tag='early_stopping')

y_pred_list = []
y_prob_list = []
y_std_list = []
correct_pred = []

label_hat = MaxProbabilityHat()
reduce_hat = EnsembleHatStd()

for data, y in tqdm(test_loader):
    ys_pred = ensemble.predict(data, device=device).to('cpu')
    y_mean, y_std, y_max = reduce_hat.predict(ys_pred, return_value=True)
    y_pred = label_hat.predict(y_mean)
    # y_pred, y_prob, y_std = get_confidence(y_means, y_stds)
    y_pred_list += [y_pred]
    y_prob_list += [y_max]
    y_std_list += [y_std]
    correct_pred += [y == y_pred]


y_pred_list = torch.cat(y_pred_list)
y_prob_list = torch.cat(y_prob_list)
y_std_list = torch.cat(y_std_list)
correct_pred = torch.cat(correct_pred)
print(y_prob_list.shape, correct_pred.shape)

print('accuracy: {}'.format(correct_pred.float().mean()))
print('correct - mean: {}, std: {}'.format(y_prob_list[correct_pred].mean(), y_std_list[correct_pred].mean()))
print('incorrect - mean: {}, std: {}'.format(y_prob_list[~correct_pred].mean(), y_std_list[~correct_pred].mean()))
eval_dict = {'accuracy': correct_pred.float().mean(),
             'correct': {'mean': y_prob_list[correct_pred].mean(), 'std': y_std_list[correct_pred].mean()},
             'correct': {'mean': y_prob_list[~correct_pred].mean(), 'std': y_std_list[~correct_pred].mean()}}

with open(f'{PROJECT_ROOT}/tmp/evaluation_dict.json') as json_file:
    json_file.write(json.dumps(eval_dict, indent=4))


# x = np.linspace(0., 1., 101)
# roc = [correct_pred[y_prob_list > t].type(torch.float).mean() for t in x]
# pickle_dump('./output/conf_C{}_H{}.pkl'.format(lambda_, H), roc)
# plt.plot(x, roc)
# plt.plot(x, x)
# plt.savefig('./output/confidence_plot_C{}_H{}.png'.format(lambda_, H))
#
# pickle_dump('./output/iterator', it)

y_mean[-1]
y_pred[-1]

torch.tensor([[1,5,3], [1,6,3]]).max(dim=0)[0]