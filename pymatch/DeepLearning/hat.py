import torch
import os
import shutil
from scipy.stats import entropy

from pymatch.utils.functional import one_hot_encoding


class Hat:
    def __call__(self, y, device='cpu'):
        raise NotImplementedError


class MaxProbabilityHat(Hat):
    def __init__(self):
        """
        Predicts the a label from a probability distribution.
        """
        super(MaxProbabilityHat, self).__init__()

    def __call__(self, y, device='cpu', return_value=False):
        if return_value:
            return y.argmax(dim=-1), y.max(dim=-1)[0]
        return y.argmax(dim=-1)


class MajorityHat(Hat):
    def __init__(self):
        super(MajorityHat, self).__init__()

    def __call__(self, y, device='cpu'):
        # @todo validate/test
        y_pred = []
        y_count = []

        label_hat = MaxProbabilityHat()
        y = label_hat.predict(y)

        for y_vote in y.transpose(0, 1):
            val, count = torch.unique(y_vote, return_counts=True)
            y_pred += [val[count.argmax()].item()]
            y_count += [count[count.argmax()] / float(len(self.learners))]  # @todo this will crash

        return torch.tensor(y_pred), torch.tensor(y_count)


class EnsembleHat(Hat):
    def __init__(self):
        """
        Reduces the ensemble predictions to a single average prediction with std
        """
        super(EnsembleHat, self).__init__()

    def __call__(self, y, device='cpu'):
        return y.mean(dim=0)


class EnsembleHatStd(Hat):
    def __init__(self):
        """
        Reduces the ensemble predictions to a single average prediction with std
        """
        super(EnsembleHatStd, self).__init__()

    def __call__(self, y, device='cpu', return_max=False):
        if return_max:
            return y.mean(dim=0), y.std(dim=0), y.max(dim=0)[0]
        return y.mean(dim=0), y.std(dim=0)


class ConfidenceBoundHat(EnsembleHatStd):
    def __init__(self, confidence_bound):
        """
        Computes the confidence bound on a given prediction.

        Args:
            confidence_bound:   confidence interval, can be a positive or negative float. If chosen positive it is the
                                upper bound, if negative it is the lower confidence bound
        """
        super().__init__()
        self.confidence_bound = confidence_bound

    def __call__(self, y, device='cpu'):
        y_mean, y_std = super().__call__(y, device)
        return y_mean + self.confidence_bound * y_std


class ConfidenceThresholdHat(ConfidenceBoundHat):
    def __init__(self, confidence_bound, threshold, garbage_class=-1, categorical_output=False):
        super().__init__(confidence_bound)
        self.threshold = threshold
        self.categorical_output = categorical_output
        self.garbage_class = garbage_class

    def __call__(self, y, device='cpu'):
        if self.garbage_class == -1:
            self.garbage_class = y.max() + 1
        y_confident = super().__call__(y, device)
        y_prob, y_class = y_confident.max(dim=-1)
        y_class[y_prob < self.threshold] = self.garbage_class
        if self.categorical_output:
            y_class = torch.zeros(len(y_class), self.garbage_class + 1).scatter_(1, y_class.view(-1, 1), 1)
        return y_class


class ThresholdHat(Hat):
    def __init__(self, threshold=.5):
        super(ThresholdHat, self).__init__()
        self.threshold = threshold

    def __call__(self, y, device='cpu'):
        y_default = (y.max(dim=1)[0] < self.threshold).float().view(-1, 1)
        return torch.cat([y, y_default], dim=1)


class EnsembleHat3Best(Hat):
    def __init__(self):
        """
        Predict the top 3 labels of a prediction by probability.
        """
        super(EnsembleHat3Best, self).__init__()

    def __call__(self, y, device='cpu'):
        probs, classes = y.sort(1, descending=True)
        probs = probs[:, :3]
        classes = classes[:, :3]
        return classes, probs


class ImageSorter(Hat):
    def __init__(self, dataloader, target_folder='./sorted_images', idx_to_class=None):
        if not isinstance(dataloader.sampler, torch.utils.data.sampler.SequentialSampler):
            raise ValueError('Data loader is not sequential. Hint: Set DataLoader(..., shuffle=False)')
        self.img_paths = list(dataloader.dataset.imgs)
        self.target_folder = target_folder

        self.idx_to_class = {v: k for k, v in dataloader.dataset.class_to_idx.items()} \
            if idx_to_class is None else idx_to_class


        # creating target folders
        for class_name in self.idx_to_class.values():
            class_path = '{}/{}'.format(self.target_folder, class_name)
            if not os.path.exists(class_path):
                os.makedirs(class_path)

    def __call__(self, y_pred, device='cpu'):
        """
        Pipeline element, that can sort images according to their label.

        Args:
            y_pred: prediction on the dataloader provided to the hat. This is assumed to be an integer indicating the
                    class label
            device: unnecessary for this pipeline element, but part of the interface

        Returns:
            the unchanged y_pred, to enable it as part of the pipeline

        """
        if len(self.img_paths) < len(y_pred):
            raise IndexError('ImageSorter ran out of images. ImageSorter can only be used once.')
        for img, label in zip(self.img_paths, y_pred):
            shutil.copy(img[0].replace('\\', '/'), '{}/{}'.format(self.target_folder, self.idx_to_class[label.item()]))
        del self.img_paths[:len(y_pred)]
        return y_pred


class EntropyHat(EnsembleHat):
    def __init__(self):
        """
        Computes the entropy over the probabilistic label distribution of an ensemble.
        """
        super().__init__()

    def __call__(self, pred, device='cpu'):
        """

        Args:
            pred:       probability distribution over classes by each model of the ensemble
            device:     not used

        Returns:
            torch tensor of entropy for label distribution over the models
        """
        actions = pred.max(-1)[1]
        ohe_actions = one_hot_encoding(actions, n_categories=4, unsqueeze=True)
        action_dist = ohe_actions.mean(0)
        action_entropy = entropy(torch.transpose(action_dist, 0, 1))

        return super(EntropyHat, self).__call__(pred), torch.tensor(action_entropy).view(-1, 1)


class InputRepeater:
    def __init__(self, n_repeats):
        self.n_repeats = n_repeats

    def __call__(self, y):
        return torch.stack(self.n_repeats * [y])


class ImageArrange:
    """
    By default images have the channels in the 'wrong' dimension. This module does nothing but permuting the
    dimensions, so that a standard image can be fed to a Conv2d-layer
    """

    def __call__(self, x):
        return x.permute(0, 3, 1, 2)


class ThompsonAggregation(EnsembleHat):
    def __call__(self, pred, device='cpu'):
        pred = torch.rand(size=(3, 5, 4))  # agent, states, actions
        idx = torch.randint(low=0, high=pred.shape[0], size=pred.shape[1:]).to(pred.device)
        idx_ohe = one_hot_encoding(idx, n_categories=pred.shape[0], unsqueeze=True)
        return (pred * torch.permute(idx_ohe, [2, 0, 1])).sum(0)
