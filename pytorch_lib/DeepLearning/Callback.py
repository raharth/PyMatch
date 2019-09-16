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
        self.img_path = '{}/learning_curve.png'.format(img_path)

    def callback(self, model, figsize=(10, 10)):
        fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=figsize)

        ax[0].plot(model.losses)
        ax[0].plot(model.val_losses)
        ax[0].legend(['train', 'val'])
        ax[0].set_title('loss')
        ax[0].set_ylabel('loss')

        ax[1].plot(model.train_accuracy)
        ax[1].plot(model.val_accuracy)
        ax[1].legend(['train', 'val'])
        ax[1].set_title('accuracy')
        ax[1].set_ylabel('accuracy in %')
        ax[1].set_xlabel('epoch')

        fig.savefig(self.img_path)
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
