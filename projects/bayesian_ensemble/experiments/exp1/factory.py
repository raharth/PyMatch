import torch.nn as nn
import torch.optim as optim
from pytorch_lib.DeepLearning.learner import ClassificationLearner
from pytorch_lib.DeepLearning.loss import AnkerLossClassification


def factory(model_class, train_loader, val_loader, device, lr, momentum, name, n_classes, C=.1, H=1., path='./tmp'):
    """ Factory for trainer object"""
    model = model_class(n_classes=n_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    crit = AnkerLossClassification(crit=nn.CrossEntropyLoss(), model=model, C=C, device=device, H=H)
    trainer = ClassificationLearner(model=model,
                                    optimizer=optimizer,
                                    crit=crit,
                                    train_loader=train_loader,
                                    val_loader=val_loader,
                                    name=name,
                                    dump_path=path)
    return trainer