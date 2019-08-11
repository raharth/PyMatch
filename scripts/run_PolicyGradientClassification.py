# general imports
import matplotlib.pyplot as plt
import numpy as np

# torch imports
import torch
import torchvision as tv
import torchvision.transforms as transforms

# own stuff
from ReinforcementLearning.Loss import REINFORCELoss
from models.Model_PG1 import Model
from ReinforcementLearning.PolicyGradientClassification import PolicyGradientClassification


plt.style.use('seaborn')

device = 'cuda'

# loading MNIST dataset
transform = transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = tv.datasets.CIFAR100(root='../../data',
                                    train=True,
                                    transform=transform,
                                    download=True,
                                    )

test_dataset = tv.datasets.CIFAR100(root='../../data',
                                    train=False,
                                    transform=transform,
                                    download=True,
                                    )

type(train_dataset.train_data)
type(train_dataset.train_labels)
train_dataset.train_data.shape

# modifying training data
n_classes = 20

# train_dataset.train_data = torch.tensor(train_dataset.train_data)
# train_dataset.train_labels = torch.tensor(train_dataset.train_labels)

mask = torch.tensor(train_dataset.train_labels) < n_classes
mask = torch.ByteTensor(mask)
train_dataset.train_data = train_dataset.train_data[mask.numpy()].shape     # todo WTF
train_dataset.train_labels = torch.tensor(train_dataset.train_labels)[mask]
train_dataset.__len__()

# Data loader
train_batch_size = 256
test_batch_size = 256

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False)

learning_rate = 1e-5
momentum = .9

agent = Model(n_classes=n_classes + 1).to(device)
optim = torch.optim.SGD(agent.parameters(), lr=learning_rate, momentum=momentum)

trainer = PolicyGradientClassification(agent=agent, optimizer=optim, train_loader=train_loader, n_classes=n_classes,
                                       crit=REINFORCELoss(), grad_clip=50., val_loader=test_loader, load_checkpoint=False)

epochs = 1

trainer.train(epochs=epochs, device=device)

trainer.validate(device)
y_pred = trainer.predict(data_loader=test_loader, device=device, prob=True)
np.unique(y_pred)

plt.plot(trainer.accuracy)
plt.show()