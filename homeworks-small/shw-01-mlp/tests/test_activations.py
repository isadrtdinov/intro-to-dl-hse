import sys
from .test_base import test_module
from torch import nn

sys.path.append('..')
import modules as mm


input_shapes = [(64, 16), (128, 32), (256, 64)]
num_tests = 100
random_seed = 4


def test_activations():
    print('test_activations ... ', end='')
    module_pairs = [
        (mm.ReLU, nn.ReLU), (mm.Sigmoid, nn.Sigmoid),
        (mm.Softmax, nn.Softmax), (mm.LogSoftmax, nn.LogSoftmax)
    ]
    for input_shape in input_shapes:
        for custom_module, torch_module in module_pairs:
            test_module(
                custom_module, torch_module, input_shape,
                outer_iters=num_tests, inner_iters=1,
                random_seed=input_shape[0] + random_seed
            )

    print('OK')
