import torch
import pymatch.DeepLearning.hat as pt_hat


hat = pt_hat.MaxProbabilityHat()
y = torch.tensor([[.2, .3, .5], [.1, .8, .1]])
y_true = torch.tensor([2, 1])
y_max_true = torch.tensor([.5, .8])
assert torch.all(hat(y) == y_true), 'MaxProbabilityHat.predict(y)'

y_pred, y_max = hat(y, return_value=True)
assert torch.all(y_pred == y_true) and torch.all(y_max == y_max_true), \
    'MaxProbabilityHat.predict(y, return_values=True)'

hat = pt_hat.EnsembleHatStd()
y = torch.tensor(
    [
        [  # pred 1
            [.7, .3, .1],
            [.7, .3, .1],
            [.7, .3, .2]
        ],
        [  # pred 2
            [.4, .6, .3],
            [.4, .6, .4],
            [.6, .4, .3]
        ],
    ]
)
y_true_mean = torch.tensor([
    [0.5500, 0.4500, 0.2000],
    [0.5500, 0.4500, 0.2500],
    [0.6500, 0.3500, 0.2500]])
y_true_std = torch.tensor([
    [0.2121, 0.2121, 0.1414],
    [0.2121, 0.2121, 0.2121],
    [0.0707, 0.0707, 0.0707]])
mean, std = hat(y)
assert (mean - y_true_mean).sum() < 1e-5 and (std - y_true_std).sum() < 1e-3

##### ConfidenceBoundHat
y = torch.tensor(
    [
        [  # pred 1
            [.7, .3, .1],
            [.7, .3, .1],
            [.7, .3, .2]
        ],
        [  # pred 2
            [.4, .6, .3],
            [.4, .6, .4],
            [.6, .4, .3]
        ],
    ]
)
ens_hat = pt_hat.EnsembleHatStd()
y_true_mean, y_true_std = ens_hat(y)

ens_hat = pt_hat.ConfidenceBoundHat(confidence_bound=-1.3)
assert (y_true_mean - 1.3 * y_true_std - ens_hat(y)).sum() < 1e-5

ens_hat = pt_hat.ConfidenceThresholdHat(confidence_bound=-1.3, threshold=.5, garbage_class=3)
assert torch.all(ens_hat(y) == torch.tensor([3, 3, 0]))

ens_hat = pt_hat.ConfidenceThresholdHat(confidence_bound=-1.3, threshold=.5, garbage_class=3, categorical_output=True)
ens_hat(y)
true_cat_output = torch.tensor([[0., 0., 0., 1.],
                                [0., 0., 0., 1.],
                                [1., 0., 0., 0.]])
assert torch.all(ens_hat(y) == true_cat_output)
