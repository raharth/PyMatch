import torch


class Predictor:

    def __init__(self, model, name):
        """

        Args:
            model: model that can predict
            name: just a name
        """
        self.model = model
        self.name = name

    def predict(self, data, device='cpu'):
        """
        Predicting a batch as tensor.

        Args:
            data: data to predict
            device: device to run the model on

        Returns:
            prediction (, true label)
        """

        with torch.no_grad():
            self.model.eval()
            self.model.to(device)
            data = data.to(device)
            y_pred = self.model.forward(data)
            return y_pred

    def load_checkpoint(self, path, tag, device='cpu'):
        """
        Loads dumped checkpoint.

        Args:
            path: source path
            tag: additional name tag

        Returns:
            None

        """
        checkpoint = torch.load(self.get_path(path=path, tag=tag), map_location=device)
        self.restore_checkpoint(checkpoint)

    def restore_checkpoint(self, checkpoint):
        """
        Restores a checkpoint_dictionary.
        This should be redefined by every derived learner (if it introduces own members), while the derived learner should call the parent function

        Args:
            checkpoint: dictionary containing the state of the learner

        Returns:
            None
        """
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def get_path(self, path, tag):
        """
        Returns the path for dumping or loading a checkpoint.

        Args:
            path: target folder
            tag: additional name tag

        Returns:

        """
        return '{}/{}_{}'.format(path, tag, self.name)