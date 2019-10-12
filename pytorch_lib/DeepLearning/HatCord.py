class HatCord:

    def __init__(self, model, hat_list):
        self.model = model
        self.hat_list = hat_list

    def predict(self, X, device='cpu', learner_args={}):
        y = self.model.predict(X, device, **learner_args)
        for hat in self.hat_list:
            y = hat.predict(y)
        return y

    def eval(self):
        self.model.eval()

    def to(self, device):
        self.model.to(device)