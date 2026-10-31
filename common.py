"""
Shared setup for the PPO files: puts the ARENA exercises directory on `sys.path` (so `gpu_env`,
`rl_utils`, `part3_ppo` etc. can be imported), and defines the device, env registry and helpers
that several modules need. Import this before any of the ARENA exercise modules.
"""

import os
import sys
import warnings
from pathlib import Path
from typing import Literal

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # don't let JAX pre-grab all GPU memory
import numpy as np
import torch as t
import torch.nn as nn

warnings.filterwarnings("ignore")

# Make sure exercises are in the path
chapter = "chapter2_rl"
section = "part3_ppo"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

from gpu_env import CartDoublePendulum, CartPole, GPUEnv, GPUProbe, MountainCar, Pendulum  # noqa: E402
from rl_utils import AtariEnvs, BraxEnvs  # noqa: E402

Arr = np.ndarray

# The networks in this section are tiny, and if you train on the CPU with
# PyTorch's default of one thread per core, threads spend longer
# coordinating than computing. Adjust if needed.
t.set_num_threads(min(4, t.get_num_threads()))

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")
ENV_DICT = {"atari": AtariEnvs, "mujoco": BraxEnvs, "classic-control": CartPole, "swing-up": CartDoublePendulum,
            "mountain-car": MountainCar, "pendulum": Pendulum, "probe": GPUProbe}
EnvType = Literal["atari", "mujoco", "classic-control", "swing-up", "mountain-car", "pendulum", "probe"]
BatchedEnvs = GPUEnv | AtariEnvs | BraxEnvs  # anything `ENV_DICT` can build; all batched & tensor-native


def layer_init(layer: nn.Linear, std=np.sqrt(2), bias_const=0.0):
    t.nn.init.orthogonal_(layer.weight, std)
    t.nn.init.constant_(layer.bias, bias_const)
    return layer
