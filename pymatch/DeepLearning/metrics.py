import numpy as np
import torch
from scipy.stats import entropy as sc_entropy


class MultipredictionEntropy:

    def __int__(self):
        """
        Computes the entropy on multiple predictions of the same batch.
        """
        super(MultipredictionEntropy, self).__init__()

    def __call__(self, y, device='cpu'):
        entr = []
        for y in torch.argmax(y, dim=-1).transpose(dim0=0, dim1=1):
         entr += [sc_entropy((np.unique(y, return_counts=True)[1] / y.shape[-1]), base=2)]
        return torch.tensor(entr)


if __name__ == '__main__':
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
            [  # pred 3
                [.4, .6, .2],
                [.6, .4, .8],
                [.6, .4, .7]
            ],
            [  # pred 4
                [.1, .9, .3],
                [.1, .9, .3],
                [.1, .9, .3]
            ]
        ]
    )
    entropy_estimation = MultipredictionEntropy()
    print(entropy_estimation(y))