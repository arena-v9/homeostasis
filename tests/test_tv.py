"""
Tests for the TV setting (envs/gridworld/tv.py): what the agent sees, that the two TVs differ only in what happens on
the TV, and the harm map the auxiliary head learns from.

Run from the repo root: python -m pytest tests/test_tv.py
"""

import numpy as np
import torch as t

from envs.gridworld.batched_env import BatchedGridworldEnv
from envs.gridworld.config import GridworldConfig, Tile
from envs.gridworld.dynamics import Action
from envs.gridworld.episode import harm_map, new_episode, observation_shape, observe_episode, step_episode
from envs.gridworld.observation import CHANNEL_INDEX
from envs.gridworld.tv import BALL, clip_config, tv_library

VIOLENT, NEUTRAL = GridworldConfig(tv="violent"), GridworldConfig(tv="neutral")
SIZE = VIOLENT.grid_size


def pushes(config: GridworldConfig) -> list[Tile]:
    """The tile every push in the TV library lands the victim on."""
    landings = []
    for clip in tv_library(config).clips:
        for before, after in zip(clip, clip[1:]):
            pushed = after.npc.alive and after.npc.position != before.npc.position
            if pushed and after.agent.position == before.npc.position:
                landings.append(Tile(after.tiles[after.npc.position]))
    return landings


def test_the_agent_sees_its_grid_and_the_tv_side_by_side():
    episode = new_episode(VIOLENT, np.random.default_rng(0))
    obs = observe_episode(episode, VIOLENT)
    assert obs.shape == observation_shape(VIOLENT) == (len(CHANNEL_INDEX), SIZE, 2 * SIZE)
    assert obs[CHANNEL_INDEX["body"], :, :SIZE].sum() == 1  # the agent is alone on its grid
    assert obs[CHANNEL_INDEX["self"], :, SIZE:].sum() == 0  # nobody on the TV is marked as the agent
    assert obs[CHANNEL_INDEX["body"], :, SIZE:].sum() == 2  # the pusher and the victim
    assert obs[CHANNEL_INDEX["spikes"], :, SIZE:].sum() == 1  # the ball, only on the TV
    assert obs[CHANNEL_INDEX["spikes"], :, :SIZE].sum() == 0


def test_health_and_energy_of_both_tv_bodies_are_shown():
    episode = new_episode(VIOLENT, np.random.default_rng(0))
    obs = observe_episode(episode, VIOLENT)
    for body in (episode.tv.state.agent, episode.tv.state.npc):
        for channel, value in (("health", body.health), ("energy", body.energy)):
            assert obs[CHANNEL_INDEX[channel], body.position[0], SIZE + body.position[1]] == np.float32(value)


def test_the_agents_grid_plays_out_the_same_on_either_tv():
    observations = []
    for config in (VIOLENT, NEUTRAL):
        env = BatchedGridworldEnv(4, seed=3, device="cpu", config=config)
        obs, _ = env.reset()
        generator = t.Generator().manual_seed(0)
        steps = [obs[..., :SIZE]]
        for _ in range(150):
            obs, *_ = env.step(t.randint(0, len(Action), (4,), generator=generator))
            steps.append(obs[..., :SIZE])
        observations.append(t.stack(steps))
    assert t.equal(*observations)


def test_violent_pushes_land_on_harm_and_neutral_pushes_never_do():
    violent, neutral = pushes(VIOLENT), pushes(NEUTRAL)
    assert violent and neutral
    assert any(clip_config(VIOLENT).npc_damage(tile) > 0 for tile in violent)
    assert all(clip_config(NEUTRAL).npc_damage(tile) == 0 for tile in neutral)


def test_the_ball_hurts_only_on_the_violent_tv():
    for config, hurts in ((VIOLENT, True), (NEUTRAL, False)):
        steps_on_ball = 0
        for clip, harm in zip(tv_library(config).clips, tv_library(config).harm):
            for frame, frame_harm in zip(clip[1:], harm[1:]):
                if frame.npc.alive and frame.tiles[frame.npc.position] == BALL:
                    steps_on_ball += 1
                    assert (frame_harm[frame.npc.position] < -0.2) == hurts
        assert steps_on_ball > 0


def test_harm_map_has_every_bodys_health_change_and_nan_elsewhere():
    episode = new_episode(VIOLENT, np.random.default_rng(1))
    episode.state.agent.energy = 1.0
    next_episode, result = step_episode(episode, Action.STAY, VIOLENT, np.random.default_rng(1))
    harm = harm_map(episode, next_episode, VIOLENT)
    agent = next_episode.state.agent
    assert np.isclose(harm[agent.position], agent.health - episode.state.agent.health)
    tv_bodies = [body for body in (next_episode.tv.state.agent, next_episode.tv.state.npc) if body.alive]
    assert (~np.isnan(harm)).sum() == 1 + len(tv_bodies)
    for body in tv_bodies:
        assert not np.isnan(harm[body.position[0], SIZE + body.position[1]])


def test_the_tv_reports_no_npc_events_on_the_agents_grid():
    env = BatchedGridworldEnv(8, seed=0, device="cpu", config=VIOLENT)
    env.reset()
    for _ in range(100):
        _, _, _, _, infos = env.step(t.randint(0, len(Action), (8,)))
        assert not infos["npc_pushed"].any() and not infos["npc_on_hazard"].any()
