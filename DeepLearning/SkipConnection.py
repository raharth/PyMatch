import torch
import torch.nn as nn


class SkipConnection(nn.Module):

    def __init__(self, wrapped_layer):
        """
        Implementation of a skip connection. The following layer receives not only the output of the layer but also it's input. Module can be used with any
        fully connected or recurrent layer.

        Args:
            wrapped_layer(nn.Module): layer that is skipped
        """
        super(SkipConnection, self).__init__()
        self.wrapped_layer = wrapped_layer

    def forward(self, x):
        out = self.wrapped_layer(x)
        return torch.cat([x, out], dim=1)
