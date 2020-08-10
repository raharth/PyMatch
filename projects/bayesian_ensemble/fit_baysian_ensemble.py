import sys
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import json
import wandb

import torch
import torchvision as tv

from pytorch_lib.DeepLearning.ensemble import BaysianEnsemble
from pytorch_lib.DeepLearning.hat import MaxProbabilityHat, EnsembleHatStd
from pytorch_lib.DeepLearning.callback import EnsembleLearningCurvePlotter
from pytorch_lib.utils.experiment import WandbExperiment, Experiment


if sys.argv[0] == '' or sys.argv[0].split('\\')[-1] == 'pydevconsole.py':
    print('Interactive')
    experiment_root = 'projects/bayesian_ensemble/experiments/exp7'
    train_script = 'projects/bayesian_ensemble/fit_baysian_ensemble.py'
else:
    print('Script mode')
    experiment_root = sys.argv[1]
    train_script = sys.argv[0]

# wandb.init(project="pytorch_test")

# experiment = WandbExperiment(root=experiment_root, param_source=experiment_root + "params.json")
experiment = Experiment(root=experiment_root)
experiment.start()
experiment.document_script(train_script)

params = experiment.get_params()
Model = experiment.get_model_class()
trainer_factory = experiment.get_factory()

use_cuda = not params['no_cuda'] and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

train_data = tv.datasets.CIFAR10('../data/CIFAR10/', train=True, transform=tv.transforms.ToTensor(), download=True)
test_data = tv.datasets.CIFAR10('../data/CIFAR10/', train=False, transform=tv.transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=params['train_batch_size'], shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=params['test_batch_size'], shuffle=False)

trainer_args = params['trainer_args']
trainer_args['train_loader'] = train_loader
trainer_args['val_loader'] = test_loader
trainer_args['device'] = device
trainer_args['path'] = trainer_args.get('path', f'{experiment.root}/tmp')


ensemble = BaysianEnsemble(model_class=Model,
                           trainer_factory=trainer_factory,
                           trainer_args=trainer_args,
                           n_model=params['n_model'],
                           callbacks=[EnsembleLearningCurvePlotter(target_folder_path=f"{experiment.root}/tmp")])

# wandb.watch(ensemble.learners[0].model)

ensemble.fit(device=device, **params['fit_params'])
experiment.finish()

ensemble.load_checkpoint(path=f'{experiment.root}/tmp/early_stopping', tag='early_stopping')

y_pred_list = []
y_prob_list = []
y_std_list = []
correct_pred = []
ys = []

label_hat = MaxProbabilityHat()
reduce_hat = EnsembleHatStd()

for data, y in tqdm(test_loader):
    ys_pred = ensemble.predict(data, device=device).to('cpu')
    y_mean, y_std = reduce_hat.predict(ys_pred)
    y_pred, y_max = label_hat.predict(y_mean, return_value=True)
    y_pred_list += [y_pred]
    y_prob_list += [y_max]
    y_std_list += [y_std]
    correct_pred += [y == y_pred]
    ys += [y]


y_pred_list = torch.cat(y_pred_list)
y_prob_list = torch.cat(y_prob_list)
y_std_list = torch.cat(y_std_list)
correct_pred = torch.cat(correct_pred)
print(y_prob_list.shape, correct_pred.shape)

print('accuracy: {}'.format(correct_pred.float().mean()))
print('correct - mean: {}, std: {}'.format(y_prob_list[correct_pred].mean(), y_std_list[correct_pred].mean()))
print('incorrect - mean: {}, std: {}'.format(y_prob_list[~correct_pred].mean(), y_std_list[~correct_pred].mean()))
eval_dict = {'accuracy': correct_pred.float().mean().item(),
             'correct': {'mean': y_prob_list[correct_pred].mean().item(), 'std': y_std_list[correct_pred].mean().item()},
             'incorrect': {'mean': y_prob_list[~correct_pred].mean().item(), 'std': y_std_list[~correct_pred].mean().item()}}

with open(f'{experiment.root}/evaluation_dict.json', 'w') as json_file:
    json_file.write(json.dumps(eval_dict, indent=4))

S = np.linspace(0., 3., 100)
P = np.linspace(.1, 1., 100)
accuracy_map = np.zeros((len(P), len(S)))
for i, p in enumerate(P):
    for j, s in enumerate(S):
        y_pred, y_max = label_hat.predict(y_mean - s * y_std, return_value=True)
        certainty_mask = y_max > p
        accuracy_map[i, j] = (y_pred[certainty_mask] == y[certainty_mask]).float().mean().item()

ticks = np.arange(0, 100, 10)
plt.imshow(accuracy_map)
plt.ylabel('threshold')
plt.xlabel('confidence')
plt.xticks(ticks, S[ticks].round(2), rotation=90)
plt.yticks(ticks, P[ticks].round(2))
plt.colorbar()
plt.tight_layout()
plt.savefig(f'{experiment.root}/threshold_map.png')
plt.close()

S = np.linspace(0., 3., 100)
curve = []
for j, s in enumerate(S):
    y_pred, y_max = label_hat.predict(y_mean - s * y_std, return_value=True)
    certainty_mask = y_max > .3
    curve += [[s, (y_pred[certainty_mask] == y[certainty_mask]).float().mean().item(), (certainty_mask).float().mean()]]
curve = np.array(curve)
plt.title('Accuracy vs. predicted datapoints')
plt.plot(curve[:, 1], curve[:, 2], '.-')
plt.xlabel('accuracy')
plt.ylabel('predicted')
# plt.show()
plt.tight_layout()
plt.savefig(f'{experiment.root}/accuracy_excluded.png')
plt.close()

plt.title('Threshold vs Prediction and Accuracy')
plt.plot(curve[:, 0], curve[:, 1], '.-', label='accuracy')
plt.plot(curve[:, 0], curve[:, 2], '.-', label='excluded')
plt.xlabel('std')
plt.ylabel('fraction/accuracy')
plt.legend()
# plt.show()
plt.tight_layout()
plt.savefig(f'{experiment.root}/std_accuracy_excluded.png')
plt.close()
