import torch


class Pipeline:

    def __init__(self, pipes, pipe_args=None, device='cpu'):
        self.pipes = pipes
        self.pipe_args = pipe_args if pipe_args is not None else [None for _ in range(len(pipes))]
        self.device = device

    def __call__(self, X):
        # for pipe, pipe_arg in zip(self.pipes, self.pipe_args):
        #     X = pipe.forward(X, device=device, **pipe_arg)
        for pipe in self.pipes:
            X = pipe(X)
        return X

    def predict_dataloader(self, data_loader, device='cpu', return_true=False):
        y_pred = []
        y_true = []

        for data, y in data_loader:
            data = data.to(device)
            y = y.to(device)

            yp = self.__call__(data)
            y_pred += [yp]
            y_true += [y]

        if return_true:
            return torch.cat(y_pred), torch.cat(y_true)
        return torch.cat(y_pred)

    def eval(self):
        for pipe in self.pipes:
            if hasattr(pipe, "eval"):
                pipe.eval()

    def to(self, device):
        self.device = device
        for pipe in self.pipes:
            if hasattr(pipe, "to"):
                pipe.to(device)

