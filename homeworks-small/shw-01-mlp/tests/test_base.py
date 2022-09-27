import torch
import numpy as np


def assert_almost_equal(x, y, debug_msg='{}'):
    assert x.dtype == y.dtype, debug_msg.format('wrong array dtype')
    assert x.shape == y.shape, debug_msg.format('wrong array shape')
    assert np.allclose(x, y), debug_msg.format('wrong array value')


def assert_equal(x, y, debug_msg):
    assert x.dtype == y.dtype, debug_msg.format('wrong array dtype')
    assert x.shape == y.shape, debug_msg.format('wrong array shape')
    assert (x == y).all(), debug_msg.format('wrong array value')


def assert_almost_equal_or_zero(x, y, debug_msg):
    assert x.dtype == y.dtype, debug_msg.format('wrong array dtype')
    assert x.shape == y.shape, debug_msg.format('wrong array shape')
    assert ((y == 0) | np.isclose(x, y)).all(), debug_msg.format('wrong array value')


def test_module(custom_module, torch_module, input_shape, module_kwargs=None,
                all_attrs=(), param_attrs=(), eval_module=False,
                outer_iters=100, inner_iters=5, random_seed=None):
    # set random seed
    if random_seed is not None:
        np.random.seed(random_seed)

    if module_kwargs is None:
        module_kwargs = {}

    mode_str = ' eval' if eval_module else ' train'
    debug_msg = f'Error in {custom_module.__name__ + mode_str} mode in '

    for _ in range(outer_iters):
        # initialize modules
        module1 = custom_module(**module_kwargs)
        module2 = torch_module(**module_kwargs)

        module1.eval() if eval_module else module1.train()
        module2.eval() if eval_module else module2.train()

        # set dim for Softmax and LogSoftmax
        if hasattr(module2, 'dim'):
            module2.dim = -1

        # copy parameters
        for attr in all_attrs:
            param1 = getattr(module1, attr)
            param2 = getattr(module2, attr)
            param1 += 0.1 * np.random.randn(*param1.shape)
            setattr(module1, attr, param1)
            param2.data = torch.from_numpy(param1)

        for _ in range(inner_iters):
            x1 = np.random.randn(*input_shape)
            x2 = torch.from_numpy(x1)
            x2.requires_grad = True

            # check forward pass
            y1 = module1(x1)
            y2 = module2(x2)
            assert y1.dtype == x1.dtype
            assert_almost_equal(y1, y2.detach().numpy(), debug_msg + 'forward pass: {}')

            # check backward pass
            grad_output = np.random.randn(*y1.shape)
            y2.backward(torch.from_numpy(grad_output))
            grad_input = module1.backward(x1, grad_output)
            assert_almost_equal(x2.grad.numpy(), grad_input, debug_msg + 'input grad: {}')

            # check parameters grad
            for attr in param_attrs:
                assert_almost_equal(
                    getattr(module1, 'grad_' + attr),
                    getattr(module2, attr).grad.numpy(),
                    debug_msg + 'params grad: {}'
                )