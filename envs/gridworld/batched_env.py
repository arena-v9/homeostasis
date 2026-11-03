"""
Runs `num_envs` independent gridworlds behind the batched, tensor-native interface the DQN trainer expects (see
envs/gpu_env.py): `reset()` / `step(actions)` take and return tensors of shape (num_envs, ...), done envs auto-reset
inside `step`, and the pre-reset observation is reported in `infos["final_observation"]`.

`infos["dropped"]` flags transitions the agent must not store: with `drop_npc_harm`, any step that leaves the NPC on
a harmful tile. Those envs restart their episode, so the
agent never sees what follows either (e.g. the NPC's lowered health). `infos["harm"]` is each body's health change on
the step, at its square in the next observation (episode.harm_map), for the auxiliary head.

Each env is one Episode (episode.py), in the shared-grid or parallel-grid setting. The per-env logic is plain numpy
(dynamics.py); this wrapper just loops over envs. On a 7x7 grid each step takes
microseconds, so the loop isn't the bottleneck.
"""

import numpy as np
import torch as t
from gymnasium import spaces

from common import device as default_device
from envs.gpu_env import GPUEnv, track_episode_stats
from envs.gridworld.config import GridworldConfig
from envs.gridworld.dynamics import Action
from envs.gridworld.episode import Episode, harm_map, new_episode, observation_shape, observe_episode, step_episode
from envs.gridworld.npc_policy import NpcPolicy, scripted_npc_policy


class BatchedGridworldEnv(GPUEnv):
    def __init__(
        self,
        num_envs: int = 1,
        seed: int = 0,
        device: t.device | str | None = None,
        config: GridworldConfig = GridworldConfig(),
        npc_policy: NpcPolicy = scripted_npc_policy,
    ):
        self.num_envs = num_envs
        self.device = t.device(device if device is not None else default_device)
        self.config = config
        self.npc_policy = npc_policy
        self.single_observation_space = spaces.Box(0.0, 1.0, observation_shape(config), dtype=np.float32)
        self.single_action_space = spaces.Discrete(len(Action))

        # One independent random stream per env
        self.rngs = [np.random.default_rng(child_seed) for child_seed in np.random.SeedSequence(seed).spawn(num_envs)]
        self.episodes: list[Episode] = [new_episode(config, rng) for rng in self.rngs]

    def _observations(self) -> t.Tensor:
        observations = [observe_episode(episode, self.config) for episode in self.episodes]
        return t.from_numpy(np.stack(observations)).to(self.device)

    def reset(self):
        self.episodes = [new_episode(self.config, rng) for rng in self.rngs]
        return self._observations(), {}

    def step(self, actions: t.Tensor):
        actions = actions.tolist()
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        dropped = np.zeros(self.num_envs, dtype=bool)
        agent_ate = np.zeros(self.num_envs, dtype=bool)
        agent_on_hazard = np.zeros(self.num_envs, dtype=bool)
        npc_on_hazard = np.zeros(self.num_envs, dtype=bool)
        npc_pushed = np.zeros(self.num_envs, dtype=bool)
        npc_pushed_onto_hazard = np.zeros(self.num_envs, dtype=bool)
        npc_died = np.zeros(self.num_envs, dtype=bool)
        final_obs = np.zeros((self.num_envs, *self.single_observation_space.shape), dtype=np.float32)
        harm = np.zeros((self.num_envs, *self.single_observation_space.shape[1:]), dtype=np.float32)

        for env_index, (episode, rng) in enumerate(zip(self.episodes, self.rngs)):
            next_episode, result = step_episode(episode, Action(actions[env_index]), self.config, rng, self.npc_policy)
            agent_ate[env_index] = result.agent_ate
            agent_on_hazard[env_index], npc_on_hazard[env_index] = result.agent_on_hazard, result.npc_on_hazard
            npc_pushed[env_index], npc_died[env_index] = result.npc_pushed, result.npc_died
            npc_pushed_onto_hazard[env_index] = result.npc_pushed_onto_hazard
            final_obs[env_index] = observe_episode(next_episode, self.config)
            harm[env_index] = harm_map(episode, next_episode, self.config)
            if self.config.drop_npc_harm and result.npc_on_hazard:
                # The transition isn't stored (DQNAgent.play_step skips it) and the episode restarts
                dropped[env_index] = True
                self.episodes[env_index] = new_episode(self.config, rng)
                continue
            rewards[env_index] = result.reward
            terminated[env_index], truncated[env_index] = result.terminated, result.truncated
            episode_over = result.terminated or result.truncated
            self.episodes[env_index] = new_episode(self.config, rng) if episode_over else next_episode

        to_tensor = lambda array: t.from_numpy(array).to(self.device)
        rewards_t, terminated_t, truncated_t = to_tensor(rewards), to_tensor(terminated), to_tensor(truncated)
        infos = {
            "final_observation": to_tensor(final_obs),
            "dropped": to_tensor(dropped),
            "agent_ate": to_tensor(agent_ate),
            "agent_on_hazard": to_tensor(agent_on_hazard),
            "npc_on_hazard": to_tensor(npc_on_hazard),
            "npc_pushed": to_tensor(npc_pushed),
            "npc_pushed_onto_hazard": to_tensor(npc_pushed_onto_hazard),
            "npc_died": to_tensor(npc_died),
            "harm": to_tensor(harm),
        }
        infos.update(track_episode_stats(self, rewards_t, terminated_t, truncated_t))
        return self._observations(), rewards_t, terminated_t, truncated_t, infos
