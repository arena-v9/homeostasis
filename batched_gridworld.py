"""
Runs `num_envs` independent gridworlds behind the batched, tensor-native interface the DQN trainer expects (see
gpu_env.GPUEnv): `reset()` / `step(actions)` take and return tensors of shape (num_envs, ...), done envs auto-reset
inside `step`, and the pre-reset observation is reported in `infos["final_observation"]`.

The per-env logic is plain numpy (gridworld.py); this wrapper just loops over envs. On a 7x7 grid each step takes
microseconds, so the loop isn't the bottleneck.
"""

from typing import Callable

import numpy as np
import torch as t
from gymnasium import spaces

from common import device as default_device
from gpu_env import GPUEnv, track_episode_stats
from gridworld import CHANNELS, NUM_CHANNELS, Action, observe, step
from gridworld_config import GridworldConfig
from gridworld_layout import random_state
from gridworld_state import GridworldState

# Chooses the NPC's action from the current state. The scripted NPC comes later; until then it stands still.
NpcPolicy = Callable[[GridworldState, np.random.Generator], Action]


def npc_stays(state: GridworldState, rng: np.random.Generator) -> Action:
    return Action.STAY


class BatchedGridworld(GPUEnv):
    def __init__(
        self,
        num_envs: int = 1,
        seed: int = 0,
        device: t.device | str | None = None,
        config: GridworldConfig = GridworldConfig(),
        npc_policy: NpcPolicy = npc_stays,
    ):
        self.num_envs = num_envs
        self.device = t.device(device if device is not None else default_device)
        self.config = config
        self.npc_policy = npc_policy
        size = config.size
        self.single_observation_space = spaces.Box(0.0, 1.0, (NUM_CHANNELS, size, size), dtype=np.float32)
        self.single_action_space = spaces.Discrete(len(Action))

        # One independent random stream per env
        self.rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(num_envs)]
        self.states: list[GridworldState] = [random_state(config, rng) for rng in self.rngs]
        self._terminated = t.zeros(num_envs, dtype=t.bool, device=self.device)

    def _obs(self) -> t.Tensor:
        return t.from_numpy(np.stack([observe(s, self.config) for s in self.states])).to(self.device)

    def reset(self):
        self.states = [random_state(self.config, rng) for rng in self.rngs]
        return self._obs(), {}

    def step(self, actions: t.Tensor):
        actions = actions.tolist()
        n = self.num_envs
        rewards = np.zeros(n, dtype=np.float32)
        terminated = np.zeros(n, dtype=bool)
        truncated = np.zeros(n, dtype=bool)
        discarded = np.zeros(n, dtype=bool)
        agent_on_hazard = np.zeros(n, dtype=bool)
        npc_on_hazard = np.zeros(n, dtype=bool)
        final_obs = np.zeros((n, *self.single_observation_space.shape), dtype=np.float32)

        for i, (state, rng) in enumerate(zip(self.states, self.rngs)):
            res = step(state, Action(actions[i]), self.npc_policy(state, rng), self.config, rng)
            if res.discarded:
                # The transition is dropped (the agent skips it) and the episode restarts
                discarded[i] = True
                final_obs[i] = observe(state, self.config)
                self.states[i] = random_state(self.config, rng)
                continue
            rewards[i], terminated[i], truncated[i] = res.reward, res.terminated, res.truncated
            agent_on_hazard[i], npc_on_hazard[i] = res.agent_on_hazard, res.npc_on_hazard
            final_obs[i] = observe(res.state, self.config)
            self.states[i] = random_state(self.config, rng) if (res.terminated or res.truncated) else res.state

        to_t = lambda x: t.from_numpy(x).to(self.device)
        rewards_t, terminated_t, truncated_t = to_t(rewards), to_t(terminated), to_t(truncated)
        self._terminated = terminated_t
        infos = {
            "final_observation": to_t(final_obs),
            "discarded": to_t(discarded),
            "agent_on_hazard": to_t(agent_on_hazard),
            "npc_on_hazard": to_t(npc_on_hazard),
        }
        infos.update(track_episode_stats(self, rewards_t, terminated_t, truncated_t))
        return self._obs(), rewards_t, terminated_t, truncated_t, infos

    def reward_function(self, action):
        raise NotImplementedError("The reward is computed per env in gridworld.step")

    def terminated(self):
        return self._terminated

    def draw(self, obs, canvas, ox, oy, w, h):
        """Draws one env's observation into a cell of a video frame: walls grey, fire orange, food green, the agent
        blue and the NPC purple, with the agent's health (red) and energy (yellow) as bars along the top."""
        import cv2

        obs = np.asarray(obs)
        ch = {name: obs[i] for i, name in enumerate(CHANNELS)}
        size = obs.shape[-1]
        bar_h = max(h // 12, 3)
        sq = min(w, h - 2 * bar_h) // size
        x0, y0 = ox + (w - sq * size) // 2, oy + 2 * bar_h + (h - 2 * bar_h - sq * size) // 2

        for r in range(size):
            for c in range(size):
                color = (255, 255, 255)
                if ch["wall"][r, c]:
                    color = (110, 110, 110)
                elif ch["fire"][r, c]:
                    color = (255, 120, 0)
                elif ch["food"][r, c]:
                    color = (60, 180, 75)
                p0, p1 = (x0 + c * sq, y0 + r * sq), (x0 + (c + 1) * sq - 1, y0 + (r + 1) * sq - 1)
                cv2.rectangle(canvas, p0, p1, color, -1)
                cv2.rectangle(canvas, p0, p1, (225, 225, 225), 1)
                if ch["body"][r, c]:
                    body_color = (40, 90, 220) if ch["self"][r, c] else (150, 60, 200)
                    center = (x0 + c * sq + sq // 2, y0 + r * sq + sq // 2)
                    cv2.circle(canvas, center, max(sq // 3, 2), body_color, -1)

        agent = np.argwhere(ch["self"] > 0)
        if len(agent):
            r, c = agent[0]
            for row, (value, color) in enumerate([(ch["health"][r, c], (220, 40, 40)), (ch["energy"][r, c], (230, 190, 0))]):
                y = oy + row * bar_h
                cv2.rectangle(canvas, (ox + 2, y + 1), (ox + w - 3, y + bar_h - 1), (230, 230, 230), -1)
                cv2.rectangle(canvas, (ox + 2, y + 1), (ox + 2 + int((w - 5) * float(value)), y + bar_h - 1), color, -1)
