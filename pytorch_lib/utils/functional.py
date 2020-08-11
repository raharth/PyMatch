import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import json
import sys


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
    return sys.argv[0] == '' or sys.argv[0].split('\\')[-1] == 'pydevconsole.py'
