from pymatch.DeepLearning.models import AutoEncoder
from pymatch.DeepLearning.learner import RegressionLearner
import torch
import torch.nn as nn
import torch.nn.functional as F
from pymatch.utils.Dataset import Dataset
import pandas as pd


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(4, 2)
        # self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        return F.relu(self.fc1(x))


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        # self.fc2 = nn.Linear(5, 10)

    def forward(self, x):
        return F.relu(self.fc1(x))

# loading data
dataset = pd.read_csv('./data/iris/iris.data.csv', names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])
dataset['species'] = pd.Categorical(dataset['species']).codes
dataset = dataset.sample(frac=1, random_state=1234)
train_input = torch.tensor(dataset.values[:120, :4]).type(torch.FloatTensor)
# train_target = dataset.values[:120, 4]
test_input = torch.tensor(dataset.values[120:, :4]).type(torch.FloatTensor)
# test_target = dataset.values[120:, 4]
train_dataset = Dataset(train_input, train_input)
test_dataset = Dataset(test_input, test_input)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=30, shuffle=False)


# create model
encoder = Encoder()
decoder = Decoder()
auto_encoder = AutoEncoder(encoder=encoder, decoder=decoder)

# create learner
crit = nn.MSELoss()
optim = torch.optim.SGD(auto_encoder.parameters(), lr=0.01, momentum=0.9)
learner = RegressionLearner(model=auto_encoder, optimizer=optim, crit=crit, train_loader=train_loader)

learner.fit(100, device='cpu')

