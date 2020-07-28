import torch


class Hat:

    def predict(self, y, device='cpu'):
        raise NotImplementedError


class DefaultClassHat(Hat):

    def __init__(self):
        """
        Adding a default class to a sigmoid output. This is adding an additional output that sums up to 1 with all
        other output nodes. This is only useful if there is no softmax output. Also is might become less useful if
        there are many possible labels.

        """
        super(DefaultClassHat, self).__init__()

    def predict(self, y, device='cpu'):
        y_def = torch.clamp(1 - y.sum(1), min=0., max=1.).view(-1, 1)
        return torch.cat([y, y_def], dim=1)


class MaxProbabilityHat(Hat):

    def __init__(self):
        """
        Predicts the a label from a probability distribution.
        """
        super(MaxProbabilityHat, self).__init__()

    def predict(self, y, device='cpu', return_value=False):
        if return_value:
            return y.argmax(dim=-1), y.max(dim=-1)[0]
        return y.argmax(dim=-1)


class MajorityHat(Hat):

    def __init__(self):
        super(MajorityHat, self).__init__()

    def predict(self, y, device='cpu'):
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

    def predict(self, y, device='cpu'):
        return y.mean(dim=1)


class EnsembleHatStd(Hat):

    def __init__(self):
        """
        Reduces the ensemble predictions to a single average prediction with std
        """
        super(EnsembleHatStd, self).__init__()

    def predict(self, y, device='cpu', return_value=False):
        if return_value:
            return y.mean(dim=0), y.std(dim=0), y.max(dim=0)[0]
        return y.mean(dim=0), y.std(dim=0)


class ThresholdHat(Hat):

    def __init__(self, threshold=.5):
        super(ThresholdHat, self).__init__()
        self.threshold = threshold

    def predict(self, y, device='cpu'):
        y_default = (y.max(dim=1)[0] < self.threshold).float().view(-1, 1)
        return torch.cat([y, y_default], dim=1)


class EnsembleHat3Best(Hat):

    def __init__(self):
        """
        Predict the top 3 labels of a prediction by probability.
        """
        super(EnsembleHat3Best, self).__init__()

    def predict(self, y, device='cpu'):
        probs, classes = y.sort(1, descending=True)
        probs = probs[:, :3]
        classes = classes[:, :3]
        return classes, probs