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


FIELDS = ("obs", "actions", "rewards", "terminated", "next_obs")


class ReplayBuffer:
    """
    Stores transitions and samples training batches from them.

    The main store holds the latest `buffer_size` transitions; once it's full, each new transition overwrites the
    oldest. Once an agent stops dying, deaths drain out of it and the agent forgets what they look like (catastrophic
    forgetting). So with `death_share` > 0, every transition that ends the episode is also copied into a small death
    store (the latest buffer_size * death_share of them), and each batch takes `death_share` of its transitions from
    there once it holds any. E.g. death_share = 0.05 puts about 6 deaths in every batch of 128.

    The `obs`, `actions`, ... attributes return the main store's transitions, oldest first.
    """

    def __init__(
        self,
        num_envs: int,
        obs_shape: tuple[int],
        action_shape: tuple[int],
        buffer_size: int,
        seed: int,
        device: t.device = device,
        death_share: float = 0.0,
    ):
        assert 0 <= death_share < 1, "death_share must be at least 0 and below 1"
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.buffer_size = buffer_size
        self.death_share = death_share
        self.device = t.device(device)
        self.rng = t.Generator(device=self.device).manual_seed(seed)
        self.main = _CircularStore(buffer_size, obs_shape, action_shape, self.device)
        self.deaths = _CircularStore(round(buffer_size * death_share), obs_shape, action_shape, self.device)

    @property
    def size(self) -> int:
        """Transitions in the main store."""
        return self.main.count

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
        new = dict(zip(FIELDS, (obs, actions, rewards, terminated, next_obs)))
        self._check(new)
        if keep is not None:
            new = {name: data[keep] for name, data in new.items()}
        self.main.add(new)
        if self.deaths.capacity:
            self.deaths.add({name: data[new["terminated"]] for name, data in new.items()})

    def _check(self, new: dict[str, Tensor]) -> None:
        """Checks shapes and dtypes (writing into the storage would silently cast a wrong dtype)."""
        expected = {
            "obs": (self.obs_shape, t.float32),
            "actions": (self.action_shape, t.int64),
            "rewards": ((), t.float32),
            "terminated": ((), t.bool),
            "next_obs": (self.obs_shape, t.float32),
        }
        for name, data in new.items():
            shape, dtype = (self.num_envs, *expected[name][0]), expected[name][1]
            assert isinstance(data, Tensor), f"`{name}` should be a tensor, got {type(data)}"
            assert data.shape == shape, f"`{name}` has shape {tuple(data.shape)}, expected {shape}"
            assert data.dtype == dtype, f"`{name}` has dtype {data.dtype}, expected {dtype}"

    def sample(self, sample_size: int, generator: t.Generator | None = None) -> ReplayBufferSamples:
        """
        Sample a batch of transitions, with replacement: `death_share` of it from the death store (once it holds any)
        and the rest from the main store. Pass a `generator` to sample without using the buffer's own random stream
        (e.g. for diagnostics that mustn't change training).
        """
        generator = generator or self.rng
        num_deaths = round(sample_size * self.death_share) if self.deaths.count else 0
        main = self.main.sample(sample_size - num_deaths, generator)
        deaths = self.deaths.sample(num_deaths, generator)
        return ReplayBufferSamples(**{name: t.cat((main[name], deaths[name])) for name in FIELDS})

    @property
    def obs(self) -> Float[Tensor, " size *obs_shape"]:
        return self.main.in_order("obs")

    @property
    def actions(self) -> Int[Tensor, " size"]:
        return self.main.in_order("actions")

    @property
    def rewards(self) -> Float[Tensor, " size"]:
        return self.main.in_order("rewards")

    @property
    def terminated(self) -> Bool[Tensor, " size"]:
        return self.main.in_order("terminated")

    @property
    def next_obs(self) -> Float[Tensor, " size *obs_shape"]:
        return self.main.in_order("next_obs")


class _CircularStore:
    """
    Fixed-size storage for transitions, allocated once: once full, each new transition overwrites the oldest, so adding
    costs the same however full it is. (ARENA's version concatenated and re-sliced the whole buffer on every add.)
    """

    def __init__(self, capacity: int, obs_shape: tuple[int], action_shape: tuple[int], device: t.device):
        self.capacity = capacity
        self.device = device
        self.data = {
            "obs": t.empty((capacity, *obs_shape), dtype=t.float32, device=device),
            "actions": t.empty((capacity, *action_shape), dtype=t.int64, device=device),
            "rewards": t.empty(capacity, dtype=t.float32, device=device),
            "terminated": t.empty(capacity, dtype=t.bool, device=device),
            "next_obs": t.empty((capacity, *obs_shape), dtype=t.float32, device=device),
        }
        self.pos = 0  # next row to write
        self.count = 0  # rows filled so far

    def add(self, new: dict[str, Tensor]) -> None:
        num_added = len(new["obs"])
        if num_added == 0:
            return
        assert num_added <= self.capacity, "can't add more transitions at once than the store holds"
        rows = (self.pos + t.arange(num_added, device=self.device)) % self.capacity
        for name, data in new.items():
            self.data[name][rows] = data
        self.pos = (self.pos + num_added) % self.capacity
        self.count = min(self.count + num_added, self.capacity)

    def sample(self, sample_size: int, generator: t.Generator) -> dict[str, Tensor]:
        """Independent uniform draws over the filled rows, which are exactly 0..count-1 until the store is full."""
        rows = t.randint(0, max(self.count, 1), (sample_size,), generator=generator, device=self.device)
        return {name: data[rows] for name, data in self.data.items()}

    def in_order(self, name: str) -> Tensor:
        """One field of the stored transitions, oldest first."""
        if self.count < self.capacity:
            return self.data[name][: self.count]
        return t.cat((self.data[name][self.pos :], self.data[name][: self.pos]))
