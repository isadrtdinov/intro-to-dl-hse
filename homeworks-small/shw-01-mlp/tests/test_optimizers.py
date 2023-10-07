import sys
import torch
import numpy as np
from .test_base import assert_almost_equal
from torch import nn

sys.path.append('..')
import modules as mm


input_shape = (256, 64)
lrs_and_wds = [(1e-3, 0), (1e-2, 1e-3), (1e-1, 1e-2)]
momenta = [0, 0.5, 0.9]
betas_list = [(0.9, 0.999), (0.8, 0.888), (0.7, 0.777)]
num_tests = 10
random_seed = 7


def test_optimizer(custom_opt, torch_opt, opt_kwargs,
                    in_features=10, out_features=20, batch_size=128,
                    outer_iters=20, inner_iters=10, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    for _ in range(outer_iters):
        module1 = mm.Sequential(
            mm.Linear(in_features, out_features),
            mm.BatchNormalization(out_features),
            mm.ReLU(),
            mm.Linear(out_features, 1)
        )
        module2 = nn.Sequential(
            nn.Linear(in_features, out_features, dtype=torch.float64),
            nn.BatchNorm1d(out_features, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(out_features, 1, dtype=torch.float64)
        )
        criterion1 = mm.MSELoss()
        criterion2 = nn.MSELoss()
        opt1 = custom_opt(module1, **opt_kwargs)
        opt2 = torch_opt(module2.parameters(), **opt_kwargs)
        debug_msg = f'Error in {custom_opt.__name__} in '

        for param1, param2 in zip(module1.parameters(), module2.parameters()):
            param2.data = torch.from_numpy(np.copy(param1))

        for _ in range(inner_iters):
            x1 = np.random.randn(batch_size, in_features)
            x2 = torch.from_numpy(x1)
            y1 = np.random.randn(batch_size, 1)
            y2 = torch.from_numpy(y1)

            opt1.zero_grad()
            out1 = module1(x1)
            module1.backward(x1, criterion1.backward(out1, y1))
            opt1.step()

            opt2.zero_grad()
            out2 = module2(x2)
            loss2 = criterion2(out2, y2)
            loss2.backward()
            opt2.step()

        for param1, param2 in zip(module1.parameters(), module2.parameters()):
            assert_almost_equal(param1, param2.detach().numpy(), debug_msg + 'params update: {}')


def test_optimizers():
    print('test_optimizers ... ', end='')
    for lr, wd in lrs_and_wds:
        for momentum in momenta:
            test_optimizer(
                mm.SGD, torch.optim.SGD,
                {'lr': lr, 'momentum': momentum, 'weight_decay': wd},
                in_features=input_shape[1], out_features=2 * input_shape[1],
                batch_size=input_shape[0], outer_iters=num_tests,
                random_seed=input_shape[0] + random_seed
            )

        for betas in betas_list:
            test_optimizer(
                mm.Adam, torch.optim.Adam,
                {'lr': lr, 'betas': betas, 'weight_decay': wd},
                in_features=input_shape[1], out_features=2 * input_shape[1],
                batch_size=input_shape[0], outer_iters=num_tests,
                random_seed=input_shape[0] + random_seed + 1
            )

    print('OK')
