from dataclasses import dataclass

import torch as t
from jaxtyping import Bool, Float, Int
from torch import Tensor

from common import device


@dataclass
class ReplayBufferSamples:
    """
    Samples from the replay buffer, as PyTorch tensors ready to feed into the Q-network.

    Data is equivalent to (s_t, a_t, r_{t+1}, d_{t+1}, s_{t+1}). Note - here, d_{t+1} is actually **terminated** rather
    than **done** (i.e. it records when the episode genuinely ended, not when it was cut off by a time limit).
    """

    obs: Float[Tensor, " sample_size *obs_shape"]
    actions: Int[Tensor, " sample_size"]
    rewards: Float[Tensor, " sample_size"]
    terminated: Bool[Tensor, " sample_size"]
    next_obs: Float[Tensor, " sample_size *obs_shape"]


class ReplayBuffer:
    """
    Contains buffer; has a method to sample from it to return a ReplayBufferSamples object.

    A circular buffer: storage is allocated once at `buffer_size` and each `add` overwrites the oldest rows, so adding
    costs the same however full the buffer is. (ARENA's version concatenated and re-sliced the whole buffer on every
    add, which copies all of it each time.) The `obs`, `actions`, ... attributes return the stored transitions in the
    order they were added, oldest first.
    """

    rng: t.Generator

    def __init__(
        self,
        num_envs: int,
        obs_shape: tuple[int],
        action_shape: tuple[int],
        buffer_size: int,
        seed: int,
        device: t.device = device,
    ):
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.buffer_size = buffer_size
        self.device = t.device(device)
        self.rng = t.Generator(device=self.device).manual_seed(seed)

        self._obs = t.empty((buffer_size, *self.obs_shape), dtype=t.float32, device=self.device)
        self._actions = t.empty((buffer_size, *self.action_shape), dtype=t.int64, device=self.device)
        self._rewards = t.empty(buffer_size, dtype=t.float32, device=self.device)
        self._terminated = t.empty(buffer_size, dtype=t.bool, device=self.device)
        self._next_obs = t.empty((buffer_size, *self.obs_shape), dtype=t.float32, device=self.device)
        self.pos = 0  # next row to write
        self.size = 0  # rows filled so far (at most buffer_size)

    def add(
        self,
        obs: Float[Tensor, " num_envs *obs_shape"],
        actions: Int[Tensor, " num_envs"],
        rewards: Float[Tensor, " num_envs"],
        terminated: Bool[Tensor, " num_envs"],
        next_obs: Float[Tensor, " num_envs *obs_shape"],
        keep: Bool[Tensor, " num_envs"] | None = None,
    ) -> None:
        """
        Add a batch of transitions to the replay buffer. If `keep` is given, only the transitions where it's True are
        added (e.g. to drop transitions the environment flagged as discarded).
        """
        # Check shapes & datatypes (writing into the storage would silently cast a wrong dtype)
        for name, data, expected_shape, expected_dtype in zip(
            ["obs", "actions", "rewards", "terminated", "next_obs"],
            [obs, actions, rewards, terminated, next_obs],
            [self.obs_shape, self.action_shape, (), (), self.obs_shape],
            [t.float32, t.int64, t.float32, t.bool, t.float32],
        ):
            assert isinstance(data, Tensor), f"`{name}` should be a tensor, got {type(data)}"
            assert data.shape == (self.num_envs, *expected_shape), f"`{name}` has shape {tuple(data.shape)}, expected {(self.num_envs, *expected_shape)}"
            assert data.dtype == expected_dtype, f"`{name}` has dtype {data.dtype}, expected {expected_dtype}"

        if keep is not None:
            obs, actions, rewards, terminated, next_obs = (
                data[keep] for data in (obs, actions, rewards, terminated, next_obs)
            )
        num_added = obs.shape[0]

        # Write over the oldest rows, wrapping around the end of the storage
        idx = (self.pos + t.arange(num_added, device=self.device)) % self.buffer_size
        self._obs[idx] = obs
        self._actions[idx] = actions
        self._rewards[idx] = rewards
        self._terminated[idx] = terminated
        self._next_obs[idx] = next_obs
        self.pos = (self.pos + num_added) % self.buffer_size
        self.size = min(self.size + num_added, self.buffer_size)

    def sample(self, sample_size: int) -> ReplayBufferSamples:
        """
        Sample a batch of transitions from the buffer, with replacement.
        """
        # Independent uniform draws over the filled rows, so duplicates are possible. Until the buffer is full,
        # the filled rows are exactly 0..size-1.
        indices = t.randint(0, self.size, (sample_size,), generator=self.rng, device=self.device)

        return ReplayBufferSamples(
            obs=self._obs[indices],                  # s_t
            actions=self._actions[indices],          # a_t
            rewards=self._rewards[indices],          # r_{t+1}
            terminated=self._terminated[indices],    # d_{t+1}: terminated (not truncated!)
            next_obs=self._next_obs[indices],        # s_{t+1}
        )

    def _in_order(self, storage: Tensor) -> Tensor:
        """The filled rows of `storage`, oldest first."""
        if self.size < self.buffer_size:
            return storage[: self.size]
        return t.cat((storage[self.pos :], storage[: self.pos]))

    @property
    def obs(self) -> Float[Tensor, " size *obs_shape"]:
        return self._in_order(self._obs)

    @property
    def actions(self) -> Int[Tensor, " size"]:
        return self._in_order(self._actions)

    @property
    def rewards(self) -> Float[Tensor, " size"]:
        return self._in_order(self._rewards)

    @property
    def terminated(self) -> Bool[Tensor, " size"]:
        return self._in_order(self._terminated)

    @property
    def next_obs(self) -> Float[Tensor, " size *obs_shape"]:
        return self._in_order(self._next_obs)
