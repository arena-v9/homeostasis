"""
Batched versions of the five standard probe environments (from Andy Jones' "Debugging RL, Without the
Agonizing Pain"; adapted from ARENA's `chapter2_rl/exercises/gpu_probe.py`). Each probe's `reset()` / `step()`
take and return tensors of shape `(num_envs, ...)`. Done envs auto-reset inside `step`, with the pre-reset
observation reported in `infos["final_observation"]`.

Expected values, given discount factor gamma (these are what the probe tests in main.py assert):

    Probe1: V([0]) = 1
    Probe2: V([x]) = x          (x is -1 or +1, drawn at episode start)
    Probe3: V([0]) = gamma, V([1]) = 1
    Probe4: V([0]) = 1, optimal policy always picks action 1
    Probe5: V = 1, optimal policy picks action == observation
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch as t
from gymnasium.spaces import Box, Discrete

from common import device


class Probe1(gym.Env):
    """
    One action, observation of [0.0], one timestep long, +1 reward.

    We expect the agent to rapidly learn that the value of the constant [0.0] observation is +1.0.
    Note we're using a continuous observation space for consistency with CartPole.
    """

    action_space: Discrete
    observation_space: Box

    def __init__(self, num_envs: int = 3, seed: int = 0, device: str | t.device = device):
        super().__init__()
        self.num_envs = num_envs
        self.device = t.device(device)

        self.observation_space = Box(np.array([0.0], dtype=np.float32), np.array([0.0], dtype=np.float32))
        self.action_space = Discrete(1)
        self.single_observation_space = self.observation_space  # vector-env style aliases
        self.single_action_space = self.action_space

        self.state = t.zeros([self.num_envs, 1], dtype=t.float32, device=self.device)

    def step(self, action: t.Tensor) -> tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor, dict]:
        reward = t.ones([self.num_envs], dtype=t.float32, device=self.device)
        terminated = t.ones([self.num_envs], dtype=t.bool, device=self.device)
        truncated = t.zeros([self.num_envs], dtype=t.bool, device=self.device)

        # Every env terminates each step; the auto-reset state is identical ([0.0]).
        infos = {"final_observation": self.state.clone()}
        return self.state.clone(), reward, terminated, truncated, infos

    def reset(self, *, seed: int | None = None, options=None) -> tuple[t.Tensor, dict]:
        self.state = t.zeros([self.num_envs, 1], dtype=t.float32, device=self.device)
        return self.state.clone(), {}


class Probe2(gym.Env):
    """
    One action, observation of [-1.0] or [+1.0] (drawn at episode start), one timestep long, reward equals
    observation.

    We expect the agent to rapidly learn the value of each observation is equal to the observation.
    """

    action_space: Discrete
    observation_space: Box

    def __init__(self, num_envs: int = 3, seed: int = 0, device: str | t.device = device):
        super().__init__()
        self.num_envs = num_envs
        self.device = t.device(device)

        self.observation_space = Box(np.array([-1.0], dtype=np.float32), np.array([+1.0], dtype=np.float32))
        self.action_space = Discrete(1)
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space

        # Seeded generator so resets are reproducible without touching torch's global RNG.
        self.gen = t.Generator(device=self.device).manual_seed(seed)
        self.reset()

    def _draw_obs(self) -> t.Tensor:
        """Each episode's observation (== reward) is +/-1, drawn uniformly at episode start."""
        rand = t.rand([self.num_envs, 1], generator=self.gen, device=self.device)
        return t.where(rand < 0.5, 1.0, -1.0)

    def step(self, action: t.Tensor) -> tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor, dict]:
        reward = self.state[:, 0].clone()  # reward == the observation the agent just acted on
        terminated = t.ones([self.num_envs], dtype=t.bool, device=self.device)
        truncated = t.zeros([self.num_envs], dtype=t.bool, device=self.device)

        final_obs = self.state.clone()
        self.state = self._draw_obs()  # auto-reset: every env draws a fresh +/-1 observation

        infos = {"final_observation": final_obs}
        return self.state.clone(), reward, terminated, truncated, infos

    def reset(self, *, seed: int | None = None, options=None) -> tuple[t.Tensor, dict]:
        if seed is not None:
            self.gen.manual_seed(seed)
        self.state = self._draw_obs()
        return self.state.clone(), {}


class Probe3(gym.Env):
    """
    One action, [0.0] then [1.0] observation, two timesteps, +1 reward at the end.

    We expect the agent to rapidly learn the discounted value of the initial observation
    (V([0]) = gamma, V([1]) = 1).
    """

    action_space: Discrete
    observation_space: Box

    def __init__(self, num_envs: int = 3, seed: int = 0, device: str | t.device = device):
        super().__init__()
        self.num_envs = num_envs
        self.device = t.device(device)

        self.observation_space = Box(np.array([0.0], dtype=np.float32), np.array([+1.0], dtype=np.float32))
        self.action_space = Discrete(1)
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space

        self.reset()

    def step(self, action: t.Tensor) -> tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor, dict]:
        self.n = self.n + 1
        first = self.n == 1  # envs that just took the first step of their episode

        reward = t.where(first, 0.0, 1.0).float()  # +1 only on the 2nd (terminal) step
        terminated = ~first
        truncated = t.zeros([self.num_envs], dtype=t.bool, device=self.device)

        final_obs = self.state.clone()
        # Mid-episode envs advance to obs [1.0]; terminated envs auto-reset to obs [0.0] with n=0.
        self.state = t.where(first.unsqueeze(-1), t.ones_like(self.state), t.zeros_like(self.state))
        self.n = t.where(terminated, t.zeros_like(self.n), self.n)

        infos = {"final_observation": final_obs}
        return self.state.clone(), reward, terminated, truncated, infos

    def reset(self, *, seed: int | None = None, options=None) -> tuple[t.Tensor, dict]:
        self.state = t.zeros([self.num_envs, 1], dtype=t.float32, device=self.device)
        self.n = t.zeros([self.num_envs], dtype=t.long, device=self.device)  # per-env step counter
        return self.state.clone(), {}


class Probe4(gym.Env):
    """
    Two actions, [0.0] observation, one timestep, reward is -1.0 or +1.0 dependent on the action.

    We expect the agent to learn to choose the +1.0 action.
    """

    action_space: Discrete
    observation_space: Box

    def __init__(self, num_envs: int = 3, seed: int = 0, device: str | t.device = device):
        super().__init__()
        self.num_envs = num_envs
        self.device = t.device(device)

        self.observation_space = Box(np.array([0.0], dtype=np.float32), np.array([0.0], dtype=np.float32))
        self.action_space = Discrete(2)
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space

        self.state = t.zeros([self.num_envs, 1], dtype=t.float32, device=self.device)

    def step(self, action: t.Tensor) -> tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor, dict]:
        reward = t.where(action.reshape(self.num_envs) == 0, -1.0, 1.0).float()
        terminated = t.ones([self.num_envs], dtype=t.bool, device=self.device)
        truncated = t.zeros([self.num_envs], dtype=t.bool, device=self.device)

        # Every env terminates each step; the auto-reset state is identical ([0.0]).
        infos = {"final_observation": self.state.clone()}
        return self.state.clone(), reward, terminated, truncated, infos

    def reset(self, *, seed: int | None = None, options=None) -> tuple[t.Tensor, dict]:
        self.state = t.zeros([self.num_envs, 1], dtype=t.float32, device=self.device)
        return self.state.clone(), {}


class Probe5(gym.Env):
    """
    Two actions, random 0/1 observation (drawn at episode start), one timestep, reward is +1 if action equals
    observation, otherwise -1.

    We expect the agent to learn to match its action to the observation.
    """

    action_space: Discrete
    observation_space: Box

    def __init__(self, num_envs: int = 3, seed: int = 0, device: str | t.device = device):
        super().__init__()
        self.num_envs = num_envs
        self.device = t.device(device)

        self.observation_space = Box(np.array([0.0], dtype=np.float32), np.array([+1.0], dtype=np.float32))
        self.action_space = Discrete(2)
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space

        self.gen = t.Generator(device=self.device).manual_seed(seed)
        self.reset()

    def _draw_obs(self) -> t.Tensor:
        """Each episode's observation is 0.0 or 1.0, drawn uniformly at episode start."""
        rand = t.rand([self.num_envs, 1], generator=self.gen, device=self.device)
        return t.where(rand < 0.5, 0.0, 1.0)

    def step(self, action: t.Tensor) -> tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor, dict]:
        # Reward +1 iff the action matches the observation the agent just acted on.
        reward = t.where(action.reshape(self.num_envs).float() == self.state[:, 0], 1.0, -1.0).float()
        terminated = t.ones([self.num_envs], dtype=t.bool, device=self.device)
        truncated = t.zeros([self.num_envs], dtype=t.bool, device=self.device)

        final_obs = self.state.clone()
        self.state = self._draw_obs()  # auto-reset: every env draws a fresh 0/1 observation

        infos = {"final_observation": final_obs}
        return self.state.clone(), reward, terminated, truncated, infos

    def reset(self, *, seed: int | None = None, options=None) -> tuple[t.Tensor, dict]:
        if seed is not None:
            self.gen.manual_seed(seed)
        self.state = self._draw_obs()
        return self.state.clone(), {}
