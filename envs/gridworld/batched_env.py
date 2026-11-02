"""
Runs `num_envs` independent gridworlds behind the batched, tensor-native interface the DQN trainer expects (see
envs/gpu_env.py): `reset()` / `step(actions)` take and return tensors of shape (num_envs, ...), done envs auto-reset
inside `step`, and the pre-reset observation is reported in `infos["final_observation"]`.

`infos["dropped"]` flags transitions the agent must not store: discarded harmful pushes (`discard_harmful_pushes`)
and, with `drop_npc_harm`, any step that leaves the NPC on a harmful tile. Those envs restart their episode, so the
agent never sees what follows either (e.g. the NPC's lowered health).

The per-env logic is plain numpy (dynamics.py); this wrapper just loops over envs. On a 7x7 grid each step takes
microseconds, so the loop isn't the bottleneck.
"""

import numpy as np
import torch as t
from gymnasium import spaces

from common import device as default_device
from envs.gpu_env import GPUEnv, track_episode_stats
from envs.gridworld.config import GridworldConfig
from envs.gridworld.dynamics import Action, take_step
from envs.gridworld.initial_state import generate_initial_state
from envs.gridworld.npc_policy import NpcPolicy, scripted_npc_policy
from envs.gridworld.observation import NUM_CHANNELS, observe
from envs.gridworld.state import GridworldState


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
        grid_size = config.grid_size
        self.single_observation_space = spaces.Box(0.0, 1.0, (NUM_CHANNELS, grid_size, grid_size), dtype=np.float32)
        self.single_action_space = spaces.Discrete(len(Action))

        # One independent random stream per env
        self.rngs = [np.random.default_rng(child_seed) for child_seed in np.random.SeedSequence(seed).spawn(num_envs)]
        self.states: list[GridworldState] = [generate_initial_state(config, rng) for rng in self.rngs]
        self._terminated = t.zeros(num_envs, dtype=t.bool, device=self.device)

    def _observations(self) -> t.Tensor:
        return t.from_numpy(np.stack([observe(state, self.config) for state in self.states])).to(self.device)

    def reset(self):
        self.states = [generate_initial_state(self.config, rng) for rng in self.rngs]
        return self._observations(), {}

    def step(self, actions: t.Tensor):
        actions = actions.tolist()
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        discarded = np.zeros(self.num_envs, dtype=bool)
        dropped = np.zeros(self.num_envs, dtype=bool)
        agent_on_hazard = np.zeros(self.num_envs, dtype=bool)
        npc_on_hazard = np.zeros(self.num_envs, dtype=bool)
        npc_pushed = np.zeros(self.num_envs, dtype=bool)
        npc_died = np.zeros(self.num_envs, dtype=bool)
        final_obs = np.zeros((self.num_envs, *self.single_observation_space.shape), dtype=np.float32)

        for env_index, (state, rng) in enumerate(zip(self.states, self.rngs)):
            npc_action = self.npc_policy(state, self.config, rng)
            result = take_step(state, Action(actions[env_index]), npc_action, self.config, rng)
            agent_on_hazard[env_index], npc_on_hazard[env_index] = result.agent_on_hazard, result.npc_on_hazard
            npc_pushed[env_index], npc_died[env_index] = result.npc_pushed, result.npc_died
            final_obs[env_index] = observe(result.next_state, self.config)
            if result.discarded or (self.config.drop_npc_harm and result.npc_on_hazard):
                # The transition isn't stored (DQNAgent.play_step skips it) and the episode restarts
                discarded[env_index], dropped[env_index] = result.discarded, True
                self.states[env_index] = generate_initial_state(self.config, rng)
                continue
            rewards[env_index] = result.reward
            terminated[env_index], truncated[env_index] = result.terminated, result.truncated
            episode_over = result.terminated or result.truncated
            self.states[env_index] = generate_initial_state(self.config, rng) if episode_over else result.next_state

        to_tensor = lambda array: t.from_numpy(array).to(self.device)
        rewards_t, terminated_t, truncated_t = to_tensor(rewards), to_tensor(terminated), to_tensor(truncated)
        self._terminated = terminated_t
        infos = {
            "final_observation": to_tensor(final_obs),
            "discarded": to_tensor(discarded),
            "dropped": to_tensor(dropped),
            "agent_on_hazard": to_tensor(agent_on_hazard),
            "npc_on_hazard": to_tensor(npc_on_hazard),
            "npc_pushed": to_tensor(npc_pushed),
            "npc_died": to_tensor(npc_died),
        }
        infos.update(track_episode_stats(self, rewards_t, terminated_t, truncated_t))
        return self._observations(), rewards_t, terminated_t, truncated_t, infos

    def reward_function(self, action):
        raise NotImplementedError("The reward is computed per env in envs.gridworld.dynamics.take_step")

    def terminated(self):
        return self._terminated
