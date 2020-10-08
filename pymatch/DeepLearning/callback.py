import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from pymatch.utils.functional import scale_confusion_matrix, sliding_window
from pymatch.utils.DataHandler import DataHandler
from pymatch.utils.exception import TerminationException

import pandas as pd
import seaborn as sn
import wandb
import numpy as np
import torch


class Callback:

    def __init__(self):
        pass

    def __call__(self, model):
        raise NotImplementedError

    def start(self, model):
        pass


class Checkpointer(Callback):

    def __init__(self, checkpoint_frequ=1):
        super().__init__()
        self.checkpoint_frequ = checkpoint_frequ

    def __call__(self, model):
        if model.train_dict['epochs_run'] % self.checkpoint_frequ == 0:
            model.dump_checkpoint()


class Validator(Callback):
    def __init__(self, data_loader, validation_int, verbose=1):
        super().__init__()
        self.data_loader = data_loader
        self.validation_int = validation_int
        self.verbose = verbose

    def __call__(self, model):
        if model.train_dict['epochs_run'] % self.validation_int == 0:
            train_mode = model.model.training
            with torch.no_grad():
                model.eval()
                model.to(model.device)
                loss = []
                accuracies = []
                for data, y in self.data_loader:
                    data = data.to(model.device)
                    y = y.to(model.device)
                    y_pred = model.model(data)
                    loss += [model.crit(y_pred, y)]

                    y_pred = y_pred.max(dim=1)[1]
                    accuracies += [(y_pred == y).float()]

                loss = torch.stack(loss).mean().item()
                model.train_dict['val_losses'] = model.train_dict.get('val_losses', []) + [loss]
                model.train_dict['val_epochs'] = model.train_dict.get('val_epochs', []) + [model.train_dict['epochs_run']]
                accuracy = torch.cat(accuracies).mean().item()
                model.train_dict['val_accuracy'] = model.train_dict.get('val_accuracy', []) + [accuracy]

            if loss < model.train_dict.get('best_val_performance', np.inf):
                model.train_dict['best_train_performance'] = loss
                model.train_dict['epochs_since_last_val_improvement'] = 0

            if self.verbose == 1:
                print('val loss: {:.4f} - val accuracy: {:.4f}'.format(loss, accuracy))
            if train_mode:  # reset to original mode
                model.train()
            return loss


class EarlyStopping(Validator):
    def __init__(self, data_loader, validation_int=1, verbose=1):
        super(EarlyStopping, self).__init__(data_loader, validation_int, verbose)

    def __call__(self, model):
        if model.train_dict['epochs_run'] % self.validation_int == 0:
            if self.verbose == 1:
                print('evaluating')
            val_loss = Validator.__call__(self, model=model)
            if val_loss < model.train_dict['best_val_performance']:
                model.train_dict['best_val_performance'] = val_loss
                model.dump_checkpoint(path=model.early_stopping_path, tag='early_stopping')


class EarlyTermination(Callback):

    def __init__(self, patience):
        super(EarlyTermination, self).__init__()
        self.patience = patience

    def __call__(self, model):
        if self.patience < model.train_dict['epochs_since_last_val_improvement']:
            raise TerminationException(f'The model did not improve for the last {self.patience} steps and is '
                                       f'therefore terminated')


class ClassificationCurvePlotter(Callback):

    def __init__(self, img_path='tmp'):
        super(ClassificationCurvePlotter, self).__init__()
        self.img_path = img_path

    def __call__(self, model, args=None, return_fig=False):
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


class RegressionCurvePlotter(Callback):

    def __init__(self, img_path='tmp'):
        super(RegressionCurvePlotter, self).__init__()
        self.img_path = img_path

    def __call__(self, model, args=None, return_fig=False):
        if args is None:
            args = {}
        if 'figsize' not in args:
            args['figsize'] = (10, 10)

        fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=args['figsize'])

        ax.plot(model.train_dict['train_losses'])
        ax.plot(model.train_dict['val_epochs'], model.train_dict['val_losses'])
        ax.legend(['train', 'val'])
        ax.set_title('loss')
        ax.set_ylabel('loss')
        ax.set_xlabel('epoch')

        if return_fig:
            return fig, ax
        img_path = '{}/learning_curve_{}.png'.format(self.img_path, model.name)
        fig.savefig(img_path)
        plt.close(fig)


class MetricPlotter(Callback):
    def __init__(self, frequency=1, metric='rewards', x=None, x_label=None, y_label=None, title=None, name=None,
                 smoothing_window=None):
        super().__init__()
        self.frequency = frequency
        self.y = metric
        self.x = x
        self.y_label = y_label if y_label is not None else 'metric'
        self.x_label = x_label if x_label is not None else 'iters'
        self.title = title if title is not None else metric
        self.name = name if name is not None else metric
        self.smoothing_window = smoothing_window

    def __call__(self, model):
        if model.train_dict['epochs_run'] % self.frequency == 0:
            if self.smoothing_window is None:
                if self.x is None:
                    plt.plot(model.train_dict[self.y])
                else:
                    plt.plot(model.train_dict[self.x], model.train_dict[self.y])
            else:
                if self.x is None:
                    plt.plot(model.train_dict[self.y], label='metric', alpha=.5)
                    plt.plot(*sliding_window(self.smoothing_window,
                                             model.train_dict[self.y]), label='smoothed')
                else:
                    plt.plot(model.train_dict[self.x], model.train_dict[self.y], label='metric', alpha=.5)
                    plt.plot(*sliding_window(self.smoothing_window,
                                             model.train_dict[self.y],
                                             index=model.train_dict.get(self.x, None)), label='smoothed')
            plt.ylabel(self.y_label)
            plt.xlabel(self.x_label)
            plt.title(self.title)
            plt.tight_layout()
            plt.savefig(f'{model.dump_path}/{self.name}.png')
            plt.close()


class SmoothedMetricPlotter(Callback):
    # @todo is this simply redundant?
    def __init__(self, metric, frequency=1, window=10,
                 x=None, x_label=None, y_label=None, title=None, name=None):
        super().__init__()
        self.frequency = frequency
        self.y = metric
        self.x = x
        self.window = window
        self.y_label = y_label if y_label is not None else 'metric'
        self.x_label = x_label if x_label is not None else 'iters'
        self.title = title if title is not None else metric
        self.name = name if name is not None else metric

    def __call__(self, model):
        if model.train_dict['epochs_run'] % self.frequency == 0:
            if self.x is None:
                plt.plot(*sliding_window(self.window,
                                         model.train_dict[self.y]))
            else:
                plt.plot(*sliding_window(self.window,
                                         model.train_dict[self.y],
                                         index=model.train_dict.get(self.x, None)))
            plt.ylabel(self.y_label)
            plt.xlabel(self.x_label)
            plt.title(f'smoothed {self.title}')
            plt.tight_layout()
            plt.savefig(f'{model.dump_path}/smoothed_{self.name}.png')
            plt.close()


class EnsembleLearningCurvePlotter(Callback):

    def __init__(self, target_folder_path='tmp'):
        """
        Plotting the learning curves of an entire ensemble
         
        Args:
            target_folder_path:   path to dump the resulting image to 
        """
        super(EnsembleLearningCurvePlotter, self).__init__()
        self.img_path = target_folder_path

    def __call__(self, ensemble, args=None, return_fig=False):
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
        fig.savefig(img_path, dpi=fig.dpi, bbox_inches='tight', pad_inches=0.5)
        plt.close(fig)


class ConfusionMatrixPlotter(Callback):

    def __init__(self, data_loader, img_path='./tmp', img_name='confusion_matrix'):
        super(ConfusionMatrixPlotter, self).__init__()
        self.data_loader = data_loader
        self.img_path = '{}/{}.png'.format(img_path, img_name)

    def __call__(self, model, classes, device='cpu', return_fig=False, title='Confusion Matrix'):
        y_pred, y_true = DataHandler.predict_data_loader(model=model, data_loader=self.data_loader, device=device, return_true=True)

        cm = scale_confusion_matrix(confusion_matrix(y_true, y_pred))
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

    def __init__(self, data_loader, folder_path='./tmp', file_name='report', mode='w'):
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

    def __call__(self, model, classes):
        y_pred, y_true = DataHandler.predict_data_loader(model, self.data_loader, return_true=True)
        report = classification_report(y_true.numpy(), y_pred.numpy(), digits=3, target_names=classes)
        self._write_report(report)

    def _write_report(self, report):
        with open(self.file_path, self.mode) as file:
            file.write(report)
            file.write('\n\n')
            file.close()


class WandbTrainDictLogger(Callback):
    def __init__(self):
        super(WandbTrainDictLogger, self).__init__()

    def __call__(self, model, args={}):
        log_dict = {}
        for k, v in model.train_dict.items():
            if isinstance(v, (list, np.ndarray)):
                log_dict[k] = v[-1]
            if isinstance(v, (int, float)):
                log_dict[k] = v
        wandb.log(log_dict)


class WandbExperiment(Callback):
    def __init__(self, wandb_args):
        super(WandbExperiment, self).__init__()
        self.wandb_args = wandb_args

    def start(self, learner):
        self.wandb_args['name'] = learner.name
        print(self.wandb_args)
        wandb.init(**self.wandb_args, reinit=True)
        wandb.watch(learner.model)

    def __call__(self, model, args={}):
        log_dict = {}
        for k, v in model.train_dict.items():
            if isinstance(v, (list, np.ndarray)):
                log_dict[k] = v[-1]
            if isinstance(v, (int, float)):
                log_dict[k] = v
        wandb.log(log_dict)
