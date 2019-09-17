import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix

from pytorch_lib.utils.Functional import scale_confusion_matrix, plot_confusion_matrix


class Callback:

    def __init__(self):
        pass

    def callback(self, model):
        raise NotImplementedError


class EarlyStopping(Callback):

    def __init__(self):
        super(EarlyStopping, self).__init__()

    def callback(self, model):
        raise NotImplementedError


class EarlyTermination(Callback):

    def __init__(self):
        super(EarlyTermination, self).__init__()

    def callback(self, model):
        raise NotImplementedError


class LearningCurvePlotter(Callback):

    def __init__(self, img_path='tmp'):
        super(LearningCurvePlotter, self).__init__()
        self.img_path = img_path

    def callback(self, model, figsize=(10, 10)):
        fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=figsize)
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

        img_path = '{}/learning_curve_{}.png'.format(self.img_path, model.name)
        fig.savefig(img_path)
        plt.close(fig)


class EnsembleLearningCurvePlotter(Callback):

    def __init__(self, img_path='tmp'):
        super(EnsembleLearningCurvePlotter, self).__init__()
        self.img_path = img_path

    def callback(self, ensemble, figsize=(10, 10)):
        fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=figsize)
        # fig.suptitle('{}'.format(model.name))
        names = []

        for learner in ensemble.learners:
            ax[0, 0].plot(learner.train_dict['train_losses'])
            names += [learner.name]
        ax[0, 0].legend(names)
        ax[0, 0].set_title('train loss')
        ax[0, 0].set_ylabel('loss')

        for learner in ensemble.learners:
            ax[1, 0].plot(learner.train_dict['val_losses'])
        ax[1, 0].legend(names)
        ax[1, 0].set_title('validation loss')
        ax[1, 0].set_ylabel('loss')

        for learner in ensemble.learners:
            ax[0, 1].plot(learner.train_dict['train_accuracy'])
        ax[0, 1].legend(names)
        ax[0, 1].set_title('train accuracy in %')
        ax[0, 1].set_ylabel('loss')

        for learner in ensemble.learners:
            ax[1, 1].plot(learner.train_dict['val_accuracy'])
        ax[1, 1].legend(names)
        ax[1, 1].set_title('validation accuracy')
        ax[1, 1].set_ylabel('validation accuracy in %')

        # ax.set_xlabel('common xlabel')
        # ax.set_ylabel('common ylabel')


        # ax[1].plot(model.train_dict['train_accuracy'])
        # ax[1].plot(model.train_dict['val_accuracy'])
        # ax[1].legend(['train', 'val'])
        # ax[1].set_title('accuracy')
        # ax[1].set_ylabel('accuracy in %')
        # ax[1].set_xlabel('epoch')

        img_path = '{}/learning_curve_ensemble.png'.format(self.img_path)
        fig.savefig(img_path)
        plt.close(fig)


class PlotConfusionMatrix(Callback):

    def __init__(self):
        super(PlotConfusionMatrix, self).__init__()

    def callback(self, model, data_loader, classes, device='cpu'):
        # @todo nasty way of doing it, definitely refactor that piece of crap, this is probably not meant to be a callback anyways
        y_true_ens = []
        y_pred_ens = []
        for X, y in data_loader:
            y_true_ens += [y]
            y_pred_ens += [model.predict(X, device=device)]
        y_true_ens = torch.cat(y_true_ens)
        y_pred_ens = torch.cat(y_pred_ens)

        cm = scale_confusion_matrix(confusion_matrix(y_true_ens, y_pred_ens))
        plot_confusion_matrix(cm, figsize=(10, 10), class_names=classes)
        plt.title('Learner performance')
        plt.show()
