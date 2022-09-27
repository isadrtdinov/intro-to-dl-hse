import numpy as np
from abc import ABC, abstractmethod
from typing import List


"""
This assignment is inspired by a similar task
from Deep Vision and Graphics course taught in YSDA
https://github.com/yandexdataschool/deep_vision_and_graphics/tree/fall21/homework01
"""


class Module(ABC):
    """
    Basic class for all neural network modules
    """
    def __init__(self):
        self.output = None
        self.training = True

    @abstractmethod
    def compute_output(self, input: np.array) -> np.array:
        """
        Compute output for forward pass, i.e. y = f(x)
        :param input: module input (x)
        :return: module output (y)
        """
        raise NotImplementedError

    @abstractmethod
    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        Compute gradient of loss w.r.t. output, i.e. dl/dx = dl/df * df/dx
        :param input: module input (x)
        :param grad_output: gradient of loss w.r.t. output (dl/df)
        :return: gradient of loss w.r.t. input (dl/dx)
        """
        raise NotImplementedError

    def update_grad_parameters(self, input: np.array, grad_output: np.array):
        """
        Update gradient of loss w.r.t. parameters, i.e. dl/dw = dl/df * df/dw
        :param input: module input (x)
        :param grad_output: gradient of loss w.r.t. output (dl/df)
        """
        pass

    def __call__(self, input: np.array) -> np.array:
        """
        Alias for 'forward' method
        :param input: module input
        :return: module output
        """
        return self.forward(input)

    def forward(self, input: np.array) -> np.array:
        """
        Forward pass through the module
        :param input: module input
        :return: module output
        """
        self.output = self.compute_output(input)
        return self.output

    def backward(self, input: np.array, grad_output: np.array) -> np.array:
        """
        Backward pass through the module
        :param input: module input
        :param grad_output: gradient of loss w.r.t. output
        :return: gradient of loss w.r.t. input
        """
        grad_input = self.compute_grad_input(input, grad_output)
        self.update_grad_parameters(input, grad_output)
        return grad_input

    def train(self):
        """
        Switch module to training mode
        """
        self.training = True

    def eval(self):
        """
        Switch module to evaluation mode
        """
        self.training = False

    def zero_grad(self):
        """
        Zero module gradients
        """
        pass

    def parameters(self) -> List[np.array]:
        """
        Get list of all trainable parameters
        """
        return []

    def parameters_grad(self) -> List[np.array]:
        """
        Get list of all parameters gradients
        """
        return []

    def __repr__(self) -> str:
        """
        Representation function for printing
        """
        return f'{self.__class__.__name__}()'


class Criterion(ABC):
    """
    Basic class for all criterions (i.e. loss functions)
    """
    def __init__(self):
        self.output = None

    @abstractmethod
    def compute_output(self, input: np.array, target: np.array) -> float:
        """
        Compute loss value, i.e. l(f, y)
        :param input: neural network predictions (f)
        :param target: ground truth targets (y)
        :return: loss value (l(f, y))
        """
        raise NotImplementedError

    @abstractmethod
    def compute_grad_input(self, input: np.array, target: np.array) -> np.array:
        """
        Compute gradient of loss w.r.t. input, i.e. dl/df
        :param input: neural network predictions (f)
        :param target: ground truth targets (y)
        :return: gradient of loss w.r.t. input (dl/df)
        """
        raise NotImplementedError

    def __call__(self, input: np.array, target: np.array) -> float:
        """
        Alias for 'forward' method
        :param input: neural network predictions
        :param target: ground truth targets
        :return: loss value
        """
        return self.forward(input, target)

    def forward(self, input: np.array, target: np.array) -> float:
        """
        Forward pass through the criterion
        :param input: neural network predictions
        :param target: ground truth targets
        :return: loss value
        """
        self.output = self.compute_output(input, target)
        return self.output

    def backward(self, input: np.array, target: np.array) -> np.array:
        """
        Backward pass through the criterion
        :param input: neural network predictions
        :param target: ground truth targets
        :return: gradient of loss w.r.t. input
        """
        grad_input = self.compute_grad_input(input, target)
        return grad_input

    def __repr__(self) -> str:
        """
        Representation function for printing
        """
        return f'{self.__class__.__name__}()'


class Optimizer(ABC):
    """
    Basic class for all optimizers
    """
    def __init__(self, module: Module):
        """
        :param module: neural network containing parameters to optimize
        """
        self.module = module
        self.state = {}  # storing current state of optimizer

    def zero_grad(self):
        """
        Zero module gradients
        """
        self.module.zero_grad()

    @abstractmethod
    def step(self):
        """
        Process one step of optimizer
        """
        raise NotImplementedError
