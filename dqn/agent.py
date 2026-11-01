import torch as t
from jaxtyping import Float, Int
from torch import Tensor

from dqn.exploration import epsilon_greedy_policy, linear_schedule
from dqn.networks import QNetwork
from dqn.replay_buffer import ReplayBuffer


class DQNAgent:
    """Base Agent class handling the interaction with the environment."""

    def __init__(
        self,
        envs,
        buffer: ReplayBuffer,
        q_network: QNetwork,
        start_e: float,
        end_e: float,
        exploration_fraction: float,
        total_timesteps: int,
        rng: t.Generator,
    ):
        self.envs = envs
        self.buffer = buffer
        self.q_network = q_network
        self.start_e = start_e
        self.end_e = end_e
        self.exploration_fraction = exploration_fraction
        self.total_timesteps = total_timesteps
        self.rng = rng

        self.step = 0  # Tracking number of environment steps taken (summed across all environments)
        self.obs, _ = self.envs.reset()  # Need a starting observation
        self.epsilon = start_e  # Starting value (will be updated in `get_actions`)

    def play_step(self) -> dict:
        """
        Carries out a single interaction step between agent & environment, and adds results to the
        replay buffer.

        Returns `infos` (a dict containing info we will log).
        """
        actions = self.get_actions(self.obs)
        next_obs, rewards, terminated, truncated, infos = self.envs.step(actions)

        # Get `true_next_obs` by finding all environments where we terminated & replacing `next_obs`
        # with the actual terminal states (the env has already auto-reset those environments)
        done = terminated | truncated
        true_next_obs = next_obs.clone()
        true_next_obs[done] = infos["final_observation"][done]

        self.buffer.add(self.obs, actions, rewards, terminated, true_next_obs)
        self.obs = next_obs

        self.step += self.envs.num_envs
        return infos

    def get_actions(self, obs: Float[Tensor, " num_envs *obs_shape"]) -> Int[Tensor, " num_envs"]:
        """
        Samples actions according to the epsilon-greedy policy using the linear schedule for epsilon.
        """
        self.epsilon = linear_schedule(
            self.step, self.start_e, self.end_e, self.exploration_fraction, self.total_timesteps
        )
        actions = epsilon_greedy_policy(self.envs, self.q_network, self.rng, obs, self.epsilon)
        assert actions.shape == (self.envs.num_envs,)
        return actions
