import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix, classification_report

from pytorch_lib.utils.Functional import scale_confusion_matrix
from pytorch_lib.utils.DataHandler import DataHandler
from pytorch_lib.DeepLearning.Hat import LabelHat, EnsembleHat

import pandas as pd
import seaborn as sn


class Callback:

    def __init__(self):
        pass

    def callback(self, model, args):
        raise NotImplementedError


class EarlyStopping(Callback):

    def __init__(self):
        super(EarlyStopping, self).__init__()

    def callback(self, model, args):
        raise NotImplementedError


class EarlyTermination(Callback):

    def __init__(self):
        super(EarlyTermination, self).__init__()

    def callback(self, model, args):
        raise NotImplementedError


class LearningCurvePlotter(Callback):

    def __init__(self, img_path='tmp'):
        super(LearningCurvePlotter, self).__init__()
        self.img_path = img_path

    def callback(self, model, args=None, return_fig=False):
        if args is None:
            args = {}
        if 'figsize' not in args:
            args['figsize'] = (10, 10)

        fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=args['figsize'])
        fig.suptitle('{}'.format(model.name))

        ax[0].plot(model.train_dict['train_losses'])
        ax[0].plot(model.train_dict['val_losses'])
        ax[0].legend(['train', 'val'])
        ax[0].set_title('loss')
        ax[0].set_ylabel('loss')

        ax[1].plot(model.train_dict['train_accuracy'])
        ax[1].plot(model.train_dict['val_accuracy'])
        ax[1].legend(['train', 'val'])
        ax[1].set_title('accuracy')
        ax[1].set_ylabel('accuracy in %')
        ax[1].set_xlabel('epoch')

        if return_fig:
            return fig, ax
        img_path = '{}/learning_curve_{}.png'.format(self.img_path, model.name)
        fig.savefig(img_path)
        plt.close(fig)


class EnsembleLearningCurvePlotter(Callback):

    def __init__(self, img_path='tmp'):
        super(EnsembleLearningCurvePlotter, self).__init__()
        self.img_path = img_path

    def callback(self, ensemble, args=None, return_fig=False):
        if args is None:
            args = {}
        if 'figsize' not in args:
            args['figsize'] = (10, 10)

        fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=args['figsize'])
        fig.text(0.5, 0.04, 'epochs', ha='center', va='center', fontsize=20)
        fig.text(0.02, 0.5, 'performance', ha='center', va='center', fontsize=20, rotation='vertical')
        fig.suptitle('Training performance', fontsize=25)

        names = []

        for learner in ensemble.learners:
            ax[0, 0].plot(learner.train_dict['train_losses'])
            names += [learner.name]
        ax[0, 0].set_ylabel('loss', fontsize=15)

        for learner in ensemble.learners:
            ax[0, 1].plot(learner.train_dict['val_epochs'], learner.train_dict['val_losses'])

        for learner in ensemble.learners:
            ax[1, 0].plot(learner.train_dict['train_accuracy'])
        ax[1, 0].set_ylabel('accuracy in %', fontsize=15)
        ax[1, 0].set_xlabel('train', fontsize=15)

        for learner in ensemble.learners:
            ax[1, 1].plot(learner.train_dict['val_epochs'], learner.train_dict['val_accuracy'])
        ax[1, 1].set_xlabel('validation', fontsize=15)

        fig.legend(names, framealpha=0.5, loc='center right')

        img_path = '{}/learning_curve_ensemble.png'.format(self.img_path)
        if return_fig:
            return fig, ax
        fig.savefig(self.img_path, dpi=fig.dpi, bbox_inches='tight', pad_inches=0.5)
        plt.close(fig)


class ConfusionMatrixPlotter(Callback):

    def __init__(self, data_loader, img_path='./tmp', img_name='confusion_matrix'):
        super(ConfusionMatrixPlotter, self).__init__()
        self.data_loader = data_loader
        self.img_path = '{}/{}.png'.format(img_path, img_name)

    def callback(self, model, classes, device='cpu', return_fig=False, title='Confusion Matrix'):
        y_true = []
        y_pred = []
        for X, y in self.data_loader:
            y_true += [y]
            y_pred += [model.predict(X, device=device)]
        y_true_ens = torch.cat(y_true)
        y_pred_ens = torch.cat(y_pred)

        ensemble_hat = EnsembleHat()
        ensemble_hat.cover(y_pred_ens)

        cm = scale_confusion_matrix(confusion_matrix(y_true_ens, y_pred_ens))
        fig, ax = self.plot_confusion_matrix(cm, figsize=(10, 10), class_names=classes)
        fig.suptitle(title, y=.95, fontsize=25)

        if return_fig:
            return fig, ax
        fig.savefig(self.img_path, dpi=fig.dpi, bbox_inches='tight', pad_inches=0.5)
        plt.close(fig)

    def plot_confusion_matrix(self, confm, class_names=None, figsize=(8, 8), heat_map_args={}):
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
        ax.set_xlabel('predicted', fontsize=15)
        ax.set_ylabel('true', fontsize=15)
        return fig, ax


class Reporter(Callback):

    def __init__(self, data_loader, folder_path='./tmp', file_name='report', mode='a+'):
        """

        Args:
            data_loader:
            folder_path:
            file_name:
            mode: mode of the writer, 'a': append, 'w': overwrite
        """
        super(Reporter, self).__init__()
        self.mode = mode
        self.data_loader = data_loader
        self.file_path = '{}/{}.txt'.format(folder_path, file_name)

    def callback(self, model, classes):
        y_pred, y_true = DataHandler.predict_data_loader(model, self.data_loader, return_true=True)
        label_hat = LabelHat()
        y_pred = label_hat.cover(y_pred)
        report = classification_report(y_true.numpy(), y_pred.numpy(), digits=3, target_names=classes)
        self._write_report(report)

    def _write_report(self, report):
        with open(self.file_path, self.mode) as file:
            file.write(report)
            file.write('\n\n')
            file.close()