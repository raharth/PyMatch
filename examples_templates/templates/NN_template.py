import torchvision as tv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.fc1 = nn.Linear(50*24*24, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        # x = F.max_pool2d(x, 2, 2)
        # print(x.shape)
        x = x.view(-1, 50*24*24)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


def train(model, train_loader, optimizer, epoch, crit, device='cpu'):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = crit(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    torch.save(model.state_dict(), "tmp.pt")
    return loss.item()


def test(model, test_loader, crit, device='cpu'):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += crit(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return test_loss.item(), correct / len(test_loader.dataset)


# training the model
no_cuda = False
use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

train_batch_size = 64
test_batch_size = 1000
epochs = 20
lr = .01
momentum = .5
log_interval = 100


train_data = tv.datasets.CIFAR10('data/CIFAR10/', train=True, transform=tv.transforms.ToTensor(), download=True)
test_data = tv.datasets.CIFAR10('data/CIFAR10/', train=False, transform=tv.transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, shuffle=True)

crit = nn.CrossEntropyLoss()
crit_test = nn.CrossEntropyLoss(reduction='sum')
model = Model()
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
loss_train = []
loss_test = []
accuracy = []

start_time = time.time()
for epoch in range(1, epochs + 1):
    ltr = train(model, train_loader, optimizer, epoch, crit, device)
    loss_train += [ltr]
    lte, acc = test(model, test_loader, crit_test, device)
    loss_test += [lte]
    accuracy += [acc]
print('time taken: {}'.format(time.time() - start_time))


torch.save(model.state_dict(), "mnist_cnn.pt")
