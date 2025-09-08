import numpy as np
import torch
import random

"""
Helper script to handle seeds.
"""


def set_seed(seed):
    """This function handles setting the seed to enable the consistent reproduction of experiments.

    Args:
        seed (int): The seed number.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
