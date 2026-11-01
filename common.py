"""
Shared setup for the DQN files: the device, CPU thread count and seeding helper that several modules need.
"""

import random
import warnings

import numpy as np
import torch as t

warnings.filterwarnings("ignore")

# The networks in this section are tiny, and if you train on the CPU with
# PyTorch's default of one thread per core, threads spend longer
# coordinating than computing. Adjust if needed.
t.set_num_threads(min(4, t.get_num_threads()))

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")


def set_global_seeds(seed):
    """Sets random seeds in several different ways (to guarantee reproducibility)"""
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    t.backends.cudnn.deterministic = True
