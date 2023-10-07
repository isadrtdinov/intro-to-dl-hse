import sys
import torch
import numpy as np
from .test_base import assert_almost_equal
from torch import nn

sys.path.append('..')
import modules as mm


input_shapes = [(64, 16), (128, 32), (256, 64)]
num_tests = 20
random_seed = 5


def _test_sequential(in_features=10, out_features=20, batch_size=128,
                     eval_module=False, outer_iters=100,
                     inner_iters=5, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    for _ in range(outer_iters):
        module1 = mm.Sequential(
            mm.Linear(in_features, out_features),
            mm.BatchNormalization(out_features),
            mm.ReLU(),
            mm.Linear(out_features, out_features),
            mm.BatchNormalization(out_features),
            mm.Sigmoid()
        )
        module2 = nn.Sequential(
            nn.Linear(in_features, out_features, dtype=torch.float64),
            nn.BatchNorm1d(out_features, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(out_features, out_features, dtype=torch.float64),
            nn.BatchNorm1d(out_features, dtype=torch.float64),
            nn.Sigmoid(),
        )

        module1.eval() if eval_module else module1.train()
        module2.eval() if eval_module else module2.train()

        mode_str = ' eval' if eval_module else ' train'
        debug_msg = f'Error in Sequential {mode_str} mode in '

        for param1, param2 in zip(module1.parameters(), module2.parameters()):
            param2.data = torch.from_numpy(param1)

        for _ in range(inner_iters):
            x1 = np.random.randn(batch_size, in_features)
            x2 = torch.from_numpy(x1)
            x2.requires_grad = True

            y1 = module1(x1)
            y2 = module2(x2)
            assert y1.dtype == x1.dtype
            assert_almost_equal(y1, y2.detach().numpy(), debug_msg + 'forward pass: {}')

            grad_output = np.random.randn(*y1.shape)
            y2.backward(torch.from_numpy(grad_output))
            grad_input = module1.backward(x1, grad_output)
            assert_almost_equal(x2.grad.numpy(), grad_input, debug_msg + 'input grad: {}')

            for grad, param in zip(module1.parameters_grad(), module2.parameters()):
                assert_almost_equal(grad, param.grad.numpy(), debug_msg + 'params grad: {}')


def test_sequential():
    print('test_sequential ... ', end='')
    for input_shape in input_shapes:
        for eval_module in (False, True):
            _test_sequential(
                input_shape[1], 2 * input_shape[1], input_shape[0],
                eval_module=eval_module, outer_iters=num_tests,
                random_seed=input_shape[0] + random_seed
            )

    print('OK')
