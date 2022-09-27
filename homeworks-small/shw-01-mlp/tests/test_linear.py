import sys
from .test_base import test_module
from torch import nn

sys.path.append('..')
import modules as mm


input_shapes = [(64, 16), (128, 32), (256, 64)]
num_tests = 50
random_seed = 1


def test_linear():
    print('test_linear ... ', end='')
    for input_shape in input_shapes:
        for bias in (True, False):
            attrs = ('weight', 'bias') if bias else ('weight', )
            module_kwargs = {
                'in_features': input_shape[1],
                'out_features': 2 * input_shape[1],
                'bias': bias
            }

            test_module(
                mm.Linear, nn.Linear, input_shape,
                module_kwargs=module_kwargs, all_attrs=attrs,
                param_attrs=attrs, eval_module=False,
                outer_iters=num_tests, random_seed=input_shape[0] + random_seed
            )

    print('OK')
