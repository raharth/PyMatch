import torch


class Hat:

    def predict(self, y):
        raise NotImplementedError


class DefaultClassHat(Hat):

    def __init__(self):
        """
        Adding a default class to a sigmoid output.

        """
        super(DefaultClassHat, self).__init__()

    def predict(self, y):
        y_def = torch.clamp(1 - y.sum(1), min=0., max=1.).view(-1,1)
        return torch.cat([y, y_def], dim=1)


class LabelHat(Hat):

    def __init__(self):
        """
        Predicts the a label from a probability distribution.

        """
        super(LabelHat, self).__init__()

    def predict(self, y):
        return y.max(dim=1)[1]


class MajorityHat(Hat):

    def __init__(self):
        super(MajorityHat, self).__init__()

    def predict(self, y):
        # @todo validate/test
        y_pred = []
        y_count = []

        label_hat = LabelHat()
        y = label_hat.predict(y)

        for y_vote in y.transpose(0, 1):
            val, count = torch.unique(y_vote, return_counts=True)
            y_pred += [val[count.argmax()].item()]
            y_count += [count[count.argmax()] / float(len(self.learners))] # @todo this will crash

        return torch.tensor(y_pred), torch.tensor(y_count)


class EnsembleHat(Hat):

    def __init__(self):
        super(EnsembleHat, self).__init__()

    def predict(self, y):
        return y.mean(dim=1)


class EnsembleHatStd(Hat):

    def __init__(self):
        super(EnsembleHatStd, self).__init__()

    def predict(self, y):
        return y.mean(dim=0), y.std(dim=0)


class ThresholdHat(Hat):

    def __init__(self, threshold=.5):
        super(ThresholdHat, self).__init__()
        self.threshold = threshold

    def predict(self, y):
        y_default = (y.max(dim=1)[0] < self.threshold).float().view(-1, 1)
        return torch.cat([y, y_default], dim=1)

