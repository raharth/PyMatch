import torch


class Predictor:

    def __init__(self, model):
        self.model = model

    def predict(self, X, device='cpu', model_args={}):
        raise NotImplementedError


class DefaultClassPredictor(Predictor):

    def __init__(self, model):
        """
        Adding a default class to a sigmoid output.

        Args:
            model: used for prediction
        """
        super(DefaultClassPredictor, self).__init__(model=model)

    def predict(self, X, device='cpu', model_args={}):
        y = self.model.predict(X, prob=True, device=device, **model_args)
        y_def = torch.clamp(1 - y.sum(1), min=0., max=1.)
        return torch.cat([y, y_def], dim=1)


class LabelPredictor(Predictor):

    def __init__(self, model):
        """
        Predicts the a label from a probability distribution.

        Args:
            model:
        """
        super(LabelPredictor, self).__init__(model=model)

    def predict(self, X, device='cpu', model_args={}):
        y_pred = self.model.predict(X, device=device, **model_args)
        return y_pred.max(dim=1)[1]


class DataLoaderPredictor(Predictor):

    def __init__(self, model):
        """
        Predicts a data loader. Could be used around any other predictor.

        Args:
            model:
        """
        super(DataLoaderPredictor, self).__init__(model=model)

    def predict(self, data_loader, device='cpu', model_args={}):
        """
        Predict a entire data loader.

        Args:
            data_loader: data to predict
            device: device to run the model on
            return_true: return the true values as well (necessary if the data loader permutates the data
            model_args: additional model argumentes for the forward pass (is that actually something one should do anyway?)

        Returns:
            predicted labels (, true labels)

        """
        y_pred = []
        for X, y in data_loader:
            y_pred += [self.model.predict(X, device=device, **model_args)]
        y_pred = torch.cat(y_pred)
        return y_pred


class DataLoaderPredictorTrue(Predictor):

    def __init__(self, model):
        """
        Predicts a data loader. Could be used around any other predictor.

        Args:
            model:
        """
        super(DataLoaderPredictorTrue, self).__init__(model=model)

    def predict(self, data_loader, device='cpu', model_args={}):
        """
        Predict a entire data loader.

        Args:
            data_loader: data to predict
            device: device to run the model on
            model_args: additional model argumentes for the forward pass (is that actually something one should do anyway?)

        Returns:
            predicted labels (, true labels)

        """
        y_pred = []
        y_true = []
        for X, y in data_loader:
            y_true += [y]
            y_pred += [self.model.predict(X, device=device, **model_args)]
        y_pred = torch.cat(y_pred)
        y_true = torch.cat(y_true)
        return y_pred, y_true