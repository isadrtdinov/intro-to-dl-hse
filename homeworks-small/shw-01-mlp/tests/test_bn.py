import sys
from .test_base import test_module
from torch import nn

sys.path.append('..')
import modules as mm


input_shapes = [(64, 16), (128, 32), (256, 64)]
momenta = [0.1, 0.5, 0.9]
num_tests = 100
random_seed = 2


def test_bn():
    print('test_bn ... ', end='')
    for momentum, input_shape in zip(momenta, input_shapes):
        for affine in (True, False):
            for eval_module in (False, True):
                all_attrs = ('running_mean', 'running_var', 'weight', 'bias') if affine else \
                    ('running_mean', 'running_var')
                param_attrs = all_attrs[2:]
                module_kwargs = {
                    'num_features': input_shape[1],
                    'affine': affine,
                    'momentum': momentum
                }

                test_module(
                    mm.BatchNormalization, nn.BatchNorm1d, input_shape,
                    module_kwargs=module_kwargs, all_attrs=all_attrs,
                    param_attrs=param_attrs, eval_module=eval_module,
                    outer_iters=num_tests, random_seed=input_shape[0] + random_seed
                )

    print('OK')
