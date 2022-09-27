import sys
import numpy as np
from .test_base import assert_equal, assert_almost_equal_or_zero

sys.path.append('..')
import modules as mm


input_shapes = [(64, 16), (128, 32), (256, 64)]
ps = [0.0, 0.1, 0.5, 0.9]
num_tests = 30
random_seed = 3


def _test_dropout(input_shape, eval_module=False, p=0.0,
                  outer_iters=100, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    mode_str = 'eval' if eval_module else 'train'
    debug_msg = f'Error in Dropout in {mode_str} mode'

    for _ in range(outer_iters):
        x = 0.01 + np.random.rand(*input_shape)
        grad_output = 0.01 + np.random.rand(*input_shape)
        module = mm.Dropout(p)
        module.eval() if eval_module else module.train()

        y = module(x)
        grad_input = module.backward(x, grad_output)

        if eval_module:
            assert_equal(x, y, debug_msg + ' in forward pass: {}')
            assert_equal(grad_output, grad_input, debug_msg + ' in input_grad: {}')

        else:
            assert_almost_equal_or_zero(x, y * (1 - p), debug_msg + ' in forward pass: {}')
            assert_almost_equal_or_zero(grad_output, grad_input * (1 - p), debug_msg + ' in input_grad: {}')
            assert ((y != 0) | (y == grad_input)).all(), debug_msg + ': forward and backward masks do not match'

            # check binomial confidence interval
            dp = 3.9 * np.sqrt(p * (1 - p) / x.size)  # 1 - a/2 norm quantile z = 3.9 for confidence 1 - a >= 0.9999
            p_y = (y == 0).mean()
            assert p - dp <= p_y <= p + dp, debug_msg + ': estimated p not in confidence interval'


def test_dropout():
    print('test_dropout ... ', end='')
    for input_shape in input_shapes:
        for p in ps:
            for eval_module in (False, True):
                _test_dropout(
                    input_shape, eval_module, p,
                    outer_iters=num_tests, random_seed=input_shape[0] + random_seed
                )

    print('OK')
