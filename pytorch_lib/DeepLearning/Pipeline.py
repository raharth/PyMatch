class Pipeline:

    def __init__(self, pipes):
        self.pipes = pipes

    def predict(self, X, device='cpu', learner_args={}):
        y = self.model.predict(X, device, **learner_args)
        for pipe in self.pipes:

            X = pipe.predict(X, **)
        return y

    def eval(self):
        self.model.eval()

    def to(self, device):
        self.model.to(device)