from pymatch.utils.functional import one_hot_encoding
import torch


test_array = torch.tensor([0,1,2,3])
encoding = one_hot_encoding(test_array)
assert (torch.eye(4) == encoding).type(torch.float).mean() == 1.

test_array = torch.tensor([0,1,2,3])
encoding = one_hot_encoding(test_array, n_categories=4)
assert (torch.eye(4) == encoding).type(torch.float).mean() == 1.

test_array = torch.tensor([0, 1, 2, 2])
encoding = one_hot_encoding(test_array, n_categories=4)
true_values = torch.eye(4)
true_values[-1, -2:] = torch.tensor([1, 0])
assert (true_values == encoding).type(torch.float).mean() == 1.
