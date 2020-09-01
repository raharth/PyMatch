import sys
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import json

import torch
import torchvision as tv

from pytorch_lib.DeepLearning.ensemble import BaysianEnsemble
from pytorch_lib.DeepLearning.hat import MaxProbabilityHat, EnsembleHatStd
from pytorch_lib.DeepLearning.callback import EnsembleLearningCurvePlotter, EarlyTermination
from pytorch_lib.utils.experiment import WandbExperiment, Experiment
from pytorch_lib.utils.functional import interactive_python_mode


def save_fig(path):
    if interactive_python_mode():
        plt.show()
    else:
        plt.tight_layout()
        plt.savefig(path)
        plt.close()


def binarize(p_list, bins):
    return np.array([((bins[i-1] < p_list) & (p_list <= bins[i])).sum().item() for i in range(1, len(bins))])


if interactive_python_mode():
    print('Interactive')
    experiment_root = 'projects/bayesian_ensemble/experiments/ensemble/exp28'
    train_script = 'projects/bayesian_ensemble/fit_bayesian_ensemble.py'
else:
    print('Script mode')
    experiment_root = sys.argv[1]
    train_script = sys.argv[0]

experiment = Experiment(root=experiment_root)

params = experiment.get_params()
Model = experiment.get_model_class()
trainer_factory = experiment.get_factory()

use_cuda = not params['no_cuda'] and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

test_data = tv.datasets.CIFAR10('../data/CIFAR10/', train=False, transform=tv.transforms.ToTensor(), download=True)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=params['test_batch_size'], shuffle=False)

trainer_args = params['trainer_args']
trainer_args['train_loader'] = None     # train_loader
trainer_args['val_loader'] = None       # val_loader
trainer_args['device'] = device
trainer_args['path'] = trainer_args.get('path', f'{experiment.root}/tmp')


ensemble = BaysianEnsemble(model_class=Model,
                           trainer_factory=trainer_factory,
                           trainer_args=trainer_args,
                           n_model=params['n_model'],
                           callbacks=[
                               EnsembleLearningCurvePlotter(target_folder_path=f"{experiment.root}/tmp")
                           ])

ensemble.load_checkpoint(path=f'{experiment.root}/tmp/early_stopping', tag='early_stopping')

y_true = []
y_out = []

label_hat = MaxProbabilityHat()
reduce_hat = EnsembleHatStd()

for data, y in tqdm(test_loader):
    y_true += [y]
    y_out += [ensemble.predict(data, device=device).to('cpu')]

y_true = torch.cat(y_true)
y_out = torch.cat(y_out, dim=1)

y_mean, y_std = reduce_hat.predict(y_out)
y_pred, y_prob = label_hat.predict(y_mean, return_value=True)

y_std_list = y_std[torch.zeros(y_std.shape).scatter_(1, y_pred.view(-1, 1), 1).bool()]
correct_pred = (y_true == y_pred)

print('accuracy: {}'.format(correct_pred.float().mean()))
print('correct - mean: {}, std: {}'.format(y_prob[correct_pred].mean(), y_std_list[correct_pred].mean()))
print('incorrect - mean: {}, std: {}'.format(y_prob[~correct_pred].mean(), y_std_list[~correct_pred].mean()))

eval_dict = {'accuracy': correct_pred.float().mean().item(),
             'correct': {'mean': y_prob[correct_pred].mean().item(),
                         'std': y_std_list[correct_pred].mean().item()},
             'incorrect': {'mean': y_prob[~correct_pred].mean().item(),
                           'std': y_std_list[~correct_pred].mean().item()},
             'known_correct': {'mean': y_prob[(y_true < 8) & correct_pred].mean().item(),
                               'std': y_std_list[(y_true < 8) & correct_pred].mean().item()},
             'known_incorrect': {'mean': y_prob[(y_true < 8) & ~correct_pred].mean().item(),
                                 'std': y_std_list[(y_true < 8) & ~correct_pred].mean().item()},
             'unkown': {'mean': y_prob[y_true > 7].mean().item(),
                        'std': y_std_list[y_true > 7].mean().item()}}

with open(f'{experiment.root}/evaluation_dict.json', 'w') as json_file:
    json_file.write(json.dumps(eval_dict, indent=4))

S = np.linspace(0., 3., 100)
P = np.linspace(.1, 1., 100)
accuracy_map = np.zeros((len(P), len(S)))

for i, p in enumerate(P):
    for j, s in enumerate(S):
        y_pred, y_max = label_hat.predict(y_mean - s * y_std, return_value=True)
        certainty_mask = y_max > p
        accuracy_map[i, j] = (y_pred[certainty_mask] == y_true[certainty_mask]).float().mean().item()

ticks = np.arange(0, 100, 10)
plt.imshow(accuracy_map)
plt.ylabel('threshold')
plt.xlabel('confidence')
plt.xticks(ticks, S[ticks].round(2), rotation=90)
plt.yticks(ticks, P[ticks].round(2))
plt.colorbar()
save_fig(f'{experiment.root}/threshold_map.png')

S = np.linspace(0., 3., 100)
curve = []
for j, s in enumerate(S):
    y_pred, y_max = label_hat.predict(y_mean - s * y_std, return_value=True)
    certainty_mask = y_max > .3
    curve += [[s, (y_pred[certainty_mask] == y_true[certainty_mask]).float().mean().item(), (certainty_mask).float().mean()]]
curve = np.array(curve)
plt.title('Accuracy vs. predicted datapoints')
plt.plot(curve[:, 1], curve[:, 2], '.-')
plt.xlabel('accuracy')
plt.ylabel('predicted')
save_fig(f'{experiment.root}/accuracy_excluded.png')

plt.title('Threshold vs Prediction and Accuracy')
plt.plot(curve[:, 0], curve[:, 1], '.-', label='accuracy')
plt.plot(curve[:, 0], curve[:, 2], '.-', label='excluded')
plt.xlabel('std')
plt.ylabel('fraction/accuracy')
plt.legend()
save_fig(f'{experiment.root}/std_accuracy_excluded.png')

bins = np.linspace(0, 1, 101)
bin_count_c = binarize(y_prob[correct_pred], bins)
bin_count_f = binarize(y_prob[~correct_pred], bins)
plt.title('Proba distribution')
plt.plot(bins[1:], bin_count_c / bin_count_c.sum(), label='correct')
plt.plot(bins[1:], bin_count_f / bin_count_f.sum(), label='incorrect')
plt.ylabel('density')
plt.xlabel('probability')
plt.legend()
save_fig(f'{experiment.root}/proba_density.png')
