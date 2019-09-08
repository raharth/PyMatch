import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn


def scale_confusion_matrix(confm):
    return (confm.transpose() / confm.sum(1)).transpose()


def plot_confusion_matrix(confm, class_names=None, figsize=(8, 8), heat_map_args={}):
    if not 'annot' in heat_map_args:
        heat_map_args['annot'] = True
    if not 'fmt' in heat_map_args:
        heat_map_args['fmt'] = '.2%'
    if not 'vmin' in heat_map_args:
        heat_map_args['vmin'] = 0.
    if not 'vmax' in heat_map_args:
        heat_map_args['vmax'] = 1.

    if class_names is None:
        class_names = ['{}'.format(i) for i in range(len(confm))]
    df_cm = pd.DataFrame(confm, index=class_names, columns=class_names)
    plt.figure(figsize=figsize)
    # fig, ax = plt.subplots(fi)
    m = sn.heatmap(df_cm, **heat_map_args) #, ax=ax)
    m.set_yticklabels(m.get_yticklabels(), rotation=45.)
    m.set_xticklabels(m.get_xticklabels(), rotation=45.)
    plt.ylim(0., len(class_names) + .5)
    return m
