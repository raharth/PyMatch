import torch
from torch.nn import functional as F


class DuelingLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, stream_features, bias=True, adjust='mean', return_Q=True, activation_F=F.relu):
        """
        Implementation of a dueling layer for RL, according to Wang et al. 2016

        http://arxiv.org/abs/1511.06581

        Args:
            in_features:        size of each input sample
            out_features:       size of each output sample
            stream_features:    size of the value and advantage stream. If presented as ``int`` it is assumed to have
                                identical size. If presented as a tuple, the first value is used af the size of the value
                                stream, the second is used as the size of the advantage stream.
            bias:               If set to ``False``, the layer will not learn an additive bias. Default: ``True``
            adjust:             Which advantage adjustment to use. This can be either ``mean`` or ``max``
            return_Q: Determines if the model
        """
        super().__init__()
        if isinstance(stream_features, int):
            stream_features = (stream_features, stream_features)
        else:
            if len(stream_features) != 2 or not isinstance(stream_features[0], int):
                raise ValueError(f"stream_features is expected to be `int` or a tuple of `int`, but was {stream_features}")
        self.in_features = in_features
        self.out_features = out_features
        self.stream_features = stream_features
        self.adjust = adjust
        self. return_Q = return_Q

        self.linear_v1 = torch.nn.Linear(in_features=in_features, out_features=stream_features[0], bias=bias)
        self.linear_v2 = torch.nn.Linear(in_features=stream_features[0], out_features=1, bias=bias)
        self.linear_a1 = torch.nn.Linear(in_features=in_features, out_features=stream_features[1], bias=bias)
        self.linear_a2 = torch.nn.Linear(in_features=stream_features[1], out_features=out_features, bias=bias)
        self.activation = activation_F

        self.reset_parameters()

    def reset_parameters(self):
        self.linear_v1.reset_parameters()
        self.linear_v2.reset_parameters()
        self.linear_a1.reset_parameters()
        self.linear_a2.reset_parameters()

    def forward(self, input):
        """
        As Wang et al. explain there is no unqiue solution if we only use ``value + advantage``, since you can shift
        both values by a constant. Therefore they suggest one of two approaches:
            1. Subtract the max of the advantage from it, so that the max advantage always is 0
            2. Subtract the mean from it

        Args:
            input:

        Returns:

        """
        value_estimation = self.linear_v2(self.activation(self.linear_v1(input)))
        advantage_estimation = self.linear_a2(self.activation(self.linear_a1(input)))

        if self.adjust == 'mean':
            adjust = advantage_estimation.mean(-1)
        else:
            adjust = advantage_estimation.max(-1)[0]

        advantage_estimation -= adjust.unsqueeze(-1)
        if self.return_Q:
            return advantage_estimation + value_estimation
        return advantage_estimation, value_estimation

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, ' \
            f'stream_features={self.stream_features}, adjust={self.adjust}, return_Q={self.return_Q}'


class with_value_function:
    def __init__(self, agent, value_function=False):
        """
        This is only meant to make the use of the value function feature easy. It should typically be used with the
        `ValueFunctionWrapper`, to wrap around any object that needs to have access to the split value- and advantage-
        function or the combined Q-function.
        Args:
            agent:
            value_function:
        """
        self.agent = agent
        self.value_function = value_function
        for var_name, obj in vars(self.agent).items():

            if isinstance(obj, DuelingLayer):
                self.layer = obj
        self.prev_state = self.layer.return_Q

    def __enter__(self):
        self.layer.return_Q = self.value_function

    def __exit__(self, *args):
        self.layer.return_Q = self.prev_state
        return False


class ValueFunctionWrapper:
    def __init__(self, module):
        """
        Just a wrapper around any object that needs either access to the Q-function or the split value- and advantage-
        function of the DuelingLayer. The only requirement for the object is that is has a `__call__` method which is
        used here.

        Args:
            module: the module one wants to wrap around
        """
        self.module = module

    def __call__(self, agent, value_function=False, *args, **kwargs):
        with with_value_function(agent, value_function=value_function):
            res = self.module(agent, *args, **kwargs)
        return res


if __name__ == '__main__':
    class Model:
        def __init__(self, layer):
            self.layer = layer


    test_in = torch.normal(2, 3, size=(10, 4))
    duel_layer = DuelingLayer(in_features=4, out_features=3, stream_features=5, return_Q=True)
    duel_layer(test_in)

    with with_value_function(Model(duel_layer), value_function=False):
        print(duel_layer(test_in))
