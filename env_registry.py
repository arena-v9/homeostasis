"""
The batched, tensor-native environments the trainer knows how to build, keyed by `env_id`. Every entry is a
constructor taking `(num_envs, seed, device)`. To train on an environment of your own, register it here:
`ENVS["MyEnv-v0"] = MyEnv`.
"""

from typing import Callable

import torch as t

from probe_envs import Probe1, Probe2, Probe3, Probe4, Probe5

ENVS: dict[str, Callable] = {
    "Probe1-v0": Probe1,
    "Probe2-v0": Probe2,
    "Probe3-v0": Probe3,
    "Probe4-v0": Probe4,
    "Probe5-v0": Probe5,
}


def make_envs(env_id: str, num_envs: int, seed: int, device: str | t.device):
    """Builds the batched environment registered under `env_id` in `ENVS`."""
    if env_id not in ENVS:
        raise ValueError(f"Environment {env_id!r} not registered. Known ids: {list(ENVS)} - add yours with `ENVS[env_id] = constructor`.")
    return ENVS[env_id](num_envs=num_envs, seed=seed, device=device)
