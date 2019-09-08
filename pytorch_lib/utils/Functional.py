import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn


def scale_confusion_matrix(confm):
    return (confm.transpose() / confm.sum(1)).transpose()


def plot_confusion_matrix(confm, class_names=None, figsize=(8, 8)):
    if class_names is None:
        class_names = ['{}'.format(i) for i in range(len(confm))]
    df_cm = pd.DataFrame(confm, index=class_names, columns=class_names)
    plt.figure(figsize=figsize)
    # fig, ax = plt.subplots(fi)
    m = sn.heatmap(df_cm, annot=True, fmt='.2%', vmin=0., vmax=1.) #, ax=ax)
    m.set_yticklabels(m.get_yticklabels(), rotation=45.)
    m.set_xticklabels(m.get_xticklabels(), rotation=45.)
    plt.ylim(0., len(class_names) + .5)
    return m