import sys
import torch
import numpy as np
from .test_base import assert_almost_equal
from torch import nn

sys.path.append('..')
import modules as mm


input_shapes = [(64, 16), (128, 32), (256, 64)]
num_tests = 200
random_seed = 6


def test_criterion(input_shape, mse=True, outer_iters=100, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    if mse:
        module1 = mm.MSELoss()
        module2 = nn.MSELoss()
        debug_msg = f'Error in MSELoss in '
    else:
        module1 = mm.CrossEntropyLoss()
        module2 = nn.CrossEntropyLoss()
        debug_msg = f'Error in CrossEntropyLoss in '

    for _ in range(outer_iters):
        x1 = np.random.randn(*input_shape)
        y1 = np.random.randn(*input_shape) if mse \
            else np.random.randint(input_shape[1], size=(input_shape[0], ))

        x2 = torch.from_numpy(x1)
        y2 = torch.from_numpy(y1)
        x2.requires_grad = True

        l1 = module1(x1, y1)
        l2 = module2(x2, y2)
        assert_almost_equal(l1, l2.detach().numpy(), debug_msg + 'forward pass: {}')

        l2.backward()
        grad_input = module1.backward(x1, y1)
        assert_almost_equal(x2.grad.numpy(), grad_input, debug_msg + 'input grad: {}')


def test_criterions():
    print(f'test_criterions ... ', end='')
    for input_shape in input_shapes:
        for mse in (True, False):
            test_criterion(
                input_shape, mse=mse, outer_iters=num_tests,
                random_seed=input_shape[0] + random_seed
            )

    print('OK')
