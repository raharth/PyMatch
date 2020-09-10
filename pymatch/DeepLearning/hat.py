import torch
import os
import shutil


class Hat:

    def __call__(self, y, device='cpu'):
        raise NotImplementedError


# class DefaultClassHat(Hat):
#
#     def __init__(self):
#         """
#         Adding a default class to a sigmoid output. This is adding an additional output that sums up to 1 with all
#         other output nodes. This is only useful if there is no softmax output. Also is might become less useful if
#         there are many possible labels.
#
#         """
#         super(DefaultClassHat, self).__init__()
#
#     def predict(self, y, device='cpu'):
#         y_def = torch.clamp(1 - y.sum(1), min=0., max=1.).view(-1, 1)
#         return torch.cat([y, y_def], dim=1)


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
        return y.mean(dim=1)


class EnsembleHatStd(Hat):

    def __init__(self):
        """
        Reduces the ensemble predictions to a single average prediction with std
        """
        super(EnsembleHatStd, self).__init__()

    def __call__(self, y, device='cpu', return_max=False):
        if return_max:
            return y.mean(dim=1), y.std(dim=1), y.max(dim=1)[0]
        return y.mean(dim=1), y.std(dim=1)


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
        y_mean, y_std = super.__call__(y, device)
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
        y_confident = super.__call__(y, device)
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
        if not isinstance(self.dataloader.sampler, torch.utils.data.sampler.SequentialSampler):
            raise ValueError('Data loader is not sequential. Hint: Set DataLoader(..., shuffle=False)')
        self.dataloader = dataloader
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
        for img, label in zip(self.dataloader.dataset.imgs, y_pred):
            shutil.copy(img[0].replace('\\', '/'), '{}/{}'.format(self.target_folder, self.idx_to_class[label]))
        return y_pred
