import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import json
import sys
import os
import numpy as np
import torch
from scipy.stats import entropy


def scale_confusion_matrix(confm):
    return (confm.transpose() / confm.sum(1)).transpose()


def plot_confusion_matrix(confm, class_names=None, figsize=(8, 8), heat_map_args={}):
    if 'annot' not in heat_map_args:
        heat_map_args['annot'] = True
    if 'fmt' not in heat_map_args:
        heat_map_args['fmt'] = '.2%'
    if 'vmin' not in heat_map_args:
        heat_map_args['vmin'] = 0.
    if 'vmax' not in heat_map_args:
        heat_map_args['vmax'] = 1.

    if class_names is None:
        class_names = ['{}'.format(i) for i in range(len(confm))]

    df_cm = pd.DataFrame(confm, index=class_names, columns=class_names)
    fig, ax = plt.subplots(figsize=figsize)
    ax = sn.heatmap(df_cm, **heat_map_args, ax=ax)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=45.)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45.)
    ax.set_ylim(0., len(class_names) + .5)
    return fig, ax


def read_json(path):
    with open(path) as json_file:
        return json.load(json_file)


def interactive_python_mode():
    is_interactive = sys.argv[0] == '' or sys.argv[0].split('\\')[-1] == 'pydevconsole.py'
    if is_interactive:
        print('interactive mode')
    else:
        print('script mode')
    return is_interactive


def shut_down(s=30):
    os.system(f"shutdown /s /t {s}")


def sliding_window(window, values, index=None, stride=1):
    if index is None:
        index = np.arange(len(values))
    means = []
    indices = []
    hw = window / 2
    cw = int(np.ceil(hw))
    fw = int(np.floor(hw))
    for i in range(fw, len(values) - fw, stride):
        indices += [index[i]]
        means += [np.mean(values[i - fw: i + cw])]
    return np.array(indices), np.array(means)


class eval_mode:
    def __init__(self, model):
        # self.training = model.model.training
        self.training = model.training
        self.model = model
        self.no_grad = torch.no_grad()

    def __enter__(self):
        self.no_grad.__enter__()
        self.model.eval()

    def __exit__(self, *args):
        if self.training:
            self.model.train()
        self.no_grad.__exit__(None, None, None)
        return False


class train_mode:
    def __init__(self, model):
        self.training = model.model.training
        self.model = model

    def __enter__(self):
        self.model.train()

    def __exit__(self, *args):
        if not self.training:
            self.model.eval()
        return False


def one_hot_encoding(categorical_data, n_categories=None, unsqueeze=False):
    if n_categories is None:
        n_categories = categorical_data.max().item() + 1
    if unsqueeze:
        encoding = torch.zeros(list(categorical_data.shape) + [n_categories], device=categorical_data.device)
        return encoding.scatter_(-1, categorical_data.unsqueeze(-1), 1)
    else:
        encoding = torch.zeros(list(categorical_data.shape[:-1]) + [n_categories], device=categorical_data.device)
        return encoding.scatter_(-1, categorical_data, 1)


def entropy_from_prob(values):
    actions = values.max(-1)[1]
    ohe_actions = one_hot_encoding(actions, n_categories=4, unsqueeze=True)
    action_dist = ohe_actions.mean(0)
    return entropy(torch.transpose(action_dist, 0, 1))


# class device:
#     def __init__(self, device, obj_list):
#         self.obj_list = obj_list
#         self.device = device
#         self.obj_dev = [dev.device for dev in obj_list]
#
#     def __enter__(self):
#         for obj in self.obj_list:
#             obj.to(device)
#
#     def __exit__(self):
#         for obj, dev in zip(self.obj_list, self.obj_dev):
#             obj.to(dev)
