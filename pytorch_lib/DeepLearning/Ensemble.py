import torch
from tqdm import tqdm


class Ensemble:

    def __init__(self, trainer_factory, n_model, trainer_args={}, callbacks=[]):
        self.learners = []
        for i in range(n_model):
            t_args = dict(trainer_args)
            t_args['name'] = trainer_args['name'] + '_{}'.format(i) if 'name' in trainer_args else '{}'.format(i)
            self.learners.append(trainer_factory(**t_args))
        self.epochs_run = 0
        self.callbacks = callbacks

        # self.losses = [] # @todo I don't think that this is actually used anywhere

    def predict(self, x, device='cpu', return_prob=True, learner_args=None):
        """
        Predicting a data tensor.

        Args:
            x (torch.tensor): data
            device: device to run the model on
            return_prob: return probability or class label
            learner_args: additional learner arguments (this may not include the 'return_prob' argument)

        Returns:
            prediction with certainty measure.
            return_prob = True:
                (Mean percentage, standard deviation of probability)
            return_prob = False:
                (Majority vote, percentage of learners voted for that label)

        """
        if learner_args is None:
            learner_args = {}

        y_preds = [leaner.predict(x, device, return_prob=return_prob, **learner_args) for leaner in self.learners]
        y_preds = torch.stack(y_preds)

        if return_prob:
            return y_preds.mean(dim=0), y_preds.std(dim=0)

        return self.majority_vote(y_preds)

    def predict_data_loader(self, data_loader, device='cpu', return_true=False, return_prob=False):
        """
        Predicting an entire torch data loader.

        Args:
            data_loader: data loader containing the data to predict
            device: device to run the model on
            return_true: return the true values of the data loader
            return_prob: return probabilistic result (majority voting if not)

        Returns:
            prediction, certainty measure (, true values)

        """
        y_pred = []
        y_cert = []
        y_true = []
        for x, y in data_loader:
            pred = self.predict(x, device=device, return_prob=return_prob)
            y_pred += [pred[0]]
            y_cert += [pred[1]]
            y_true += [y]

        y_pred = torch.cat(y_pred)
        y_cert = torch.cat(y_cert)

        if return_true:
            y_true = torch.cat(y_true)
            return y_pred_m, y_pred_certainty, y_true

        return y_pred_m, y_pred_certainty

    def train(self, epochs, device, checkpoint_int=10, validation_int=10, restore_early_stopping=False, verbose=1, callback_iter=-1):
        """
        Trains each learner of the ensemble for a number of epochs

        Args:
            callback_iter: number of iterations till the ensemble callbacks are called
            epochs: number of epochs to train each learner
            device: device to run the models on
            checkpoint_int: every checkpoint_int iterations the model is checkpointed
            validation_int:  every validation_int iterations the model is validated
            restore_early_stopping: restores the best performing weights after training
            verbose: verbosity

        Returns:
            None

        """
        if callback_iter > 0:   # dive into a sequence of shorter training runs
            epoch_iter = [callback_iter for _ in range(epochs//callback_iter)]
            if epochs % callback_iter > 0:
                callback_iter += [epochs % callback_iter]
        else:
            epoch_iter = [epochs]

        for run_epochs in epoch_iter:
            for trainer in self.learners:
                if verbose == 1:
                    print('Trainer {}'.format(trainer.name))
                trainer.train(epochs=run_epochs, device=device, checkpoint_int=checkpoint_int,
                              validation_int=validation_int, restore_early_stopping=restore_early_stopping)
            for cb in self.callbacks:
                cb.callback(self)

    def dump_checkpoint(self, path=None, tag='checkpoint'):
        """
        Dumps a checkpoint for each learner.

        Args:
            path: source folder of the checkpoints
            tag: addition tags of the checkpoints

        Returns:
            None
        """
        for trainer in self.learners:
            trainer.dump_checkpoint(path=path, tag=tag)

    def load_checkpoint(self, path=None, tag='checkpoint'):
        """
        Loads a set of checkpoints, one for each learner

        Args:
            path: source folder of the checkpoints
            tag: addition tags of the checkpoints

        Returns:
            None

        """
        for learner in self.learners:
            learner.load_checkpoint(path=path, tag=tag)

    def run_validation(self, device='cpu'):
        """
        Validates the ensemble on the validation dataset.

        Args:
            device: device to run the evaluation on

        Returns:
            predicted and true labels for each learner

        """
        y_pred_learners = []
        y_true_learners = []

        for learner in self.learners:
            y_pred = []
            y_true = []
            for X, y in learner.val_loader:
                y_true += [y]
                y_pred += [learner.predict(X, device=device)]
            y_pred_learners += [torch.cat(y_pred)]
            y_true_learners += [torch.cat(y_true)]
        return y_pred_learners, y_true_learners

    def train_upto(self, epochs, device='cpu', checkpoint_int=10, validation_int=10, restore_early_stopping=False, verbose=1):
        """
        The primarily purpose of this method is to return to training after an interrupted trainings cycle of the ensemble.

        Args:
            epochs: max number of epochs to run the learners for
            device: device to run the models on
            checkpoint_int: every checkpoint_int iterations the model is checkpointed
            validation_int:  every validation_int iterations the model is validated
            restore_early_stopping: restores the best performing weights after training
            verbose: verbosity

        Returns:
            None

        """
        for learner in self.learners:
            train_epochs = epochs - learner.epochs_run
            if verbose == 1:
                print('Trainer {} - train for {} epochs'.format(learner.name, train_epochs))
            learner.train(epochs=train_epochs, device=device, checkpoint_int=checkpoint_int,
                          validation_int=validation_int, restore_early_stopping=restore_early_stopping)

    def majority_vote(self, y_vote):
        """
        Evaluates the predictions of the ensemble as a majority voting.

        Args:
            y_vote: probabilistic output of the ensemble as a torch tensor of the predictions, where dim=0 are the learners

        Returns:
            prediction and percentage of that prediction

        """
        y_pred = []
        y_count = []

        for y in y_vote.transpose(0, 1):
            val, count = torch.unique(y, return_counts=True)
            y_pred += [val[count.argmax()].item()]
            y_count += [count[count.argmax()] / float(len(self.learners))]
        return torch.tensor(y_pred), torch.tensor(y_count)



class BaysianEnsemble(Ensemble):

    def __init__(self, trainer_factory, n_model, trainer_args={}):
        super(BaysianEnsemble, self).__init__(trainer_factory, n_model, trainer_args=trainer_args)

    def predict(self, x, device='cpu'):
        with torch.no_grad():
            y_preds = torch.stack([trainer.model.forward(x, device=device, train=False) for trainer in self.learners])
            return y_preds.mean(dim=0).to('cpu'), y_preds.std(dim=0).to('cpu')

    def predict_class(self, x, device):
        y_pred, _ = self.predict(x, device)
        return torch.max(y_pred.data, 1)[1].data

    def predict_class_single_models(self, data_loader, device):
        for i, trainer in enumerate(self.learners):
            y_pred_list = []
            correct_pred = []

            for data, y in tqdm(data_loader):
                y_pred = trainer.predict(data, device=device, return_prob=False)
                y_pred_list += [y_pred]
                correct_pred += [y == y_pred.to('cpu')]

            y_pred_list = torch.cat(y_pred_list)
            correct_pred = torch.cat(correct_pred)

            print('{}: accuracy: {}'.format(i, correct_pred.float().mean()))
        # @todo no return?

    @staticmethod
    def get_confidence(y_mean, y_std):
        y_prob, y_pred = torch.max(y_mean, 1)
        y_confidence = []
        for y_p, y_s in zip(y_pred, y_std):
            y_confidence += [y_s[y_p]]
        return y_pred, y_prob, torch.stack(y_confidence)


