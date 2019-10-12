class Pipeline:

    def __init__(self, pipes, pipe_args=None):
        self.pipes = pipes
        self.pipe_args = pipe_args if pipe_args is not None else [None for _ in range(len(pipes))]

    def predict(self, X):
        for pipe, pipe_arg in zip(self.pipes, self.pipe_args):
            X = pipe.predict(X, **pipe_arg)
        return X

    # def eval(self):
    #     self.model.eval()
    #
    # def to(self, device):
    #     self.model.to(device)
