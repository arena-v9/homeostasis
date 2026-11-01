"""
Base class and episode-statistics helper for batched, tensor-native environments (adapted from ARENA's
`chapter2_rl/exercises/gpu_env.py`). Every env's `reset()` / `step()` take and return tensors of shape
`(num_envs, ...)` on the env's device, and done envs auto-reset inside `step`, with the pre-reset observation
reported in `infos["final_observation"]`.
"""

import time
from abc import ABC, abstractmethod

import torch
from gymnasium import spaces


def track_episode_stats(env, reward, terminated, truncated) -> dict:
    """Accumulate per-env episode return/length (on-device) and, on steps where at least one env
    finished, return a `{"final_info": [{"episode": {"r", "l", "t"}}]}` dict, averaged over the
    envs that finished this step. This is the format `get_episode_data_from_infos` (in dqn_trainer.py)
    consumes, so episode reward/length show up on the training progress bar and in wandb. The counters
    are lazily attached to the env on first call."""
    done = terminated.bool() | truncated.bool()
    if not hasattr(env, "_ep_return"):
        env._ep_return = torch.zeros_like(reward, dtype=torch.float)
        env._ep_length = torch.zeros_like(reward, dtype=torch.float)
        env._t0 = time.time()  # episode start times are stored relative to this, in float32 (no float64 on MPS)
        env._ep_start = torch.zeros(env.num_envs, device=reward.device)
    env._ep_return = env._ep_return + reward.float()
    env._ep_length = env._ep_length + 1
    if not done.any():  # the one host sync per step
        return {}
    # Everything below stays on-device: no boolean-mask indexing (which forces a device->host sync to size the
    # result) and no .cpu() copies. The returned r/l/t are 0-dim device tensors; consumers call .item().
    now = time.time() - env._t0
    n_done = done.sum()
    donef = done.float()
    episode = {
        "r": (env._ep_return * donef).sum() / n_done,
        "l": (env._ep_length * donef).sum() / n_done,
        "t": now - (env._ep_start * donef).sum() / n_done,
    }
    env._ep_return = torch.where(done, torch.zeros_like(env._ep_return), env._ep_return)
    env._ep_length = torch.where(done, torch.zeros_like(env._ep_length), env._ep_length)
    env._ep_start = torch.where(done, torch.full_like(env._ep_start, now), env._ep_start)
    return {"final_info": [{"episode": episode}]}


class GPUEnv(ABC):
    """Abstract base for a fully-on-device, batched vector environment.

    It defines just the slice of the gymnasium vector-env API that our DQN actually uses -
    `reset()` / `step()` returning tensors, plus `num_envs`, `single_observation_space` and
    `single_action_space` - so these environments do NOT need to inherit from `gym.Env`.

    `reward_function` and `terminated` are separate methods that `step` calls, rather than being
    inlined into `step`, so the reward is easy to override in a subclass without touching the dynamics.
    """

    num_envs: int
    single_observation_space: spaces.Space
    single_action_space: spaces.Space

    @abstractmethod
    def reset(self):
        """Reset all environments; return (obs, info), obs a (num_envs, *obs_shape) tensor."""

    @abstractmethod
    def step(self, action):
        """Advance one step; return (obs, reward, terminated, truncated, info) as tensors."""

    @abstractmethod
    def reward_function(self, action):
        """Per-step reward, shape (num_envs,). Override this to substitute a shaped reward."""

    @abstractmethod
    def terminated(self):
        """Per-step termination flags, shape (num_envs,) bool, computed from the current state."""

    # --- optional ---
    def draw(self, obs, canvas, ox, oy, w, h):
        """OPTIONAL per-cell renderer for the 4x4 grid video (see video.py). Given one environment's `obs`
        and a cell rectangle (ox, oy, w, h) on a uint8 RGB `canvas` (an H x W x 3 numpy array), draw that
        env's current state into the cell. An env without `draw()` simply can't be rendered to video."""
        raise NotImplementedError(f"{type(self).__name__} does not define a draw() renderer")

    def close(self):
        pass
