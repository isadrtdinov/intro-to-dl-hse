import torch
import numpy as np
import random


def set_random_seed(seed: int):
    """
    Fix random generators seed.

    Args:
        seed (int): random seed

    """
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
