import torch as t
import torch.nn as nn
from torch.distributions.categorical import Categorical

from common import BatchedEnvs
from replay_memory import ReplayMemory
from replay_minibatch import ReplayMinibatch


class PPOAgent:
    critic: nn.Sequential
    actor: nn.Sequential

    def __init__(
        self,
        envs: BatchedEnvs,
        actor: nn.Module,
        critic: nn.Module,
        memory: ReplayMemory,
    ):
        super().__init__()
        self.envs = envs
        self.actor = actor
        self.critic = critic
        self.memory = memory

        self.step = 0  # Tracking number of steps taken (across all environments)
        self.next_obs = t.as_tensor(envs.reset()[0], dtype=t.float)  # need starting obs (already on the env's device)
        self.next_terminated = t.zeros(envs.num_envs, device=self.next_obs.device, dtype=t.bool)  # need starting termination=False

    def play_step(self) -> list[dict]:
        """
        Carries out a single interaction step between the agent and the environment, and adds
        results to the replay memory.

        Returns the list of info dicts returned from `self.envs.step`.
        """
        # Get newest observations (i.e. where we're starting from)
        obs = self.next_obs
        terminated = self.next_terminated

        # Compute logits based on newest observation, and use it to get an action distribution we sample from.
        with t.inference_mode():
            logits = self.actor(obs)
        dist = Categorical(logits=logits)
        actions = dist.sample()

        next_obs, rewards, next_terminated, next_truncated, infos = self.envs.step(actions)

        # Calculate logprobs and values, and add this all to replay memory
        logprobs = dist.log_prob(actions)
        with t.inference_mode():
            values = self.critic(obs).flatten()
        self.memory.add(obs, actions, logprobs, values, rewards, terminated)

        self.next_obs = next_obs
        self.next_terminated = next_terminated

        self.step += self.envs.num_envs
        return infos

    def get_minibatches(self, gamma: float, gae_lambda: float) -> list[ReplayMinibatch]:
        """
        Gets minibatches from the replay memory, and resets the memory
        """
        with t.inference_mode():
            next_value = self.critic(self.next_obs).flatten()
        minibatches = self.memory.get_minibatches(next_value, self.next_terminated, gamma, gae_lambda)
        self.memory.reset()
        return minibatches
