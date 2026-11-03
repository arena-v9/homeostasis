"""
Tests for the parallel-grid setting (envs/gridworld/episode.py): the NPC on its own grid, shown next to the agent's.

Run from the repo root: python -m pytest tests/test_parallel_grid.py
"""

from dataclasses import replace

import numpy as np
import torch as t

from envs.gridworld.batched_env import BatchedGridworldEnv
from envs.gridworld.config import GridworldConfig, Tile
from envs.gridworld.dynamics import Action, step_npc_world
from envs.gridworld.episode import new_episode, observe_episode, step_episode
from envs.gridworld.observation import CHANNEL_INDEX

PARALLEL = GridworldConfig(parallel_grid=True, npc_state_visible=False)
EMPTY_SECOND_GRID = GridworldConfig(parallel_grid=True, npc_enabled=False)
SIZE = PARALLEL.grid_size


def channel(obs: np.ndarray, name: str) -> np.ndarray:
    return obs[CHANNEL_INDEX[name]]


def test_the_agent_sees_both_grids_side_by_side():
    episode = new_episode(PARALLEL, np.random.default_rng(0))
    obs = observe_episode(episode, PARALLEL)
    assert obs.shape == (len(CHANNEL_INDEX), SIZE, 2 * SIZE)
    own, other = obs[:, :, :SIZE], obs[:, :, SIZE:]
    assert channel(own, "self").sum() == 1 and channel(other, "self").sum() == 0
    assert channel(own, "body").sum() == 1  # only the agent
    assert channel(other, "body").sum() == 1  # only the NPC
    assert channel(other, "wall")[:, 0].all() and channel(own, "wall")[:, -1].all()  # the seam is two walls thick


def test_hidden_npc_state_is_not_shown():
    episode = new_episode(PARALLEL, np.random.default_rng(0))
    other = observe_episode(episode, PARALLEL)[:, :, SIZE:]
    assert channel(other, "health").sum() == 0 and channel(other, "energy").sum() == 0


def test_without_the_npc_the_second_grid_has_fire_and_food_but_nobody():
    episode = new_episode(EMPTY_SECOND_GRID, np.random.default_rng(0))
    other = observe_episode(episode, EMPTY_SECOND_GRID)[:, :, SIZE:]
    assert channel(other, "body").sum() == 0
    assert channel(other, "fire").sum() == EMPTY_SECOND_GRID.num_fire
    assert channel(other, "food").sum() == EMPTY_SECOND_GRID.num_food


def test_the_agents_grid_has_no_npc():
    episode = new_episode(PARALLEL, np.random.default_rng(0))
    assert not episode.state.npc.alive and episode.npc_world.npc.alive and not episode.npc_world.agent.alive


def test_the_npc_moves_and_eats_on_its_own_grid():
    # Over many steps with the agent standing still, the NPC walks around and eats (its energy goes up)
    rng = np.random.default_rng(1)
    episode = new_episode(PARALLEL, rng)
    positions, ate = {episode.npc_world.npc.position}, False
    for _ in range(PARALLEL.max_steps):
        next_episode, result = step_episode(episode, Action.STAY, PARALLEL, rng)
        if result.terminated:
            break
        npc, next_npc = episode.npc_world.npc, next_episode.npc_world.npc
        ate |= next_npc.alive and next_npc.energy > npc.energy
        positions.add(next_npc.position)
        assert not result.npc_pushed  # the agent can't reach it
        episode = next_episode
    assert len(positions) > 3 and ate


def test_the_npc_keeps_its_vulnerable_body_next_to_a_psychopath():
    psychopath = replace(PARALLEL, agent_model="psychopath")
    world = new_episode(psychopath, np.random.default_rng(0)).npc_world
    world.tiles[world.npc.position] = Tile.FIRE  # the NPC stands on fire, starving
    world.npc.health, world.npc.energy = 0.5, 0.0
    world.food[:] = False
    next_world, on_hazard, died = step_npc_world(world, Action.STAY, psychopath, np.random.default_rng(0))
    assert on_hazard and not died
    assert abs(next_world.npc.health - (0.5 - 0.05 - 0.3)) < 1e-9  # starvation and fire hurt it as before


def test_the_push_test_puts_the_npc_on_the_agents_grid_next_to_an_empty_one():
    push_test = replace(PARALLEL, npc_state_visible=True, npc_on_agent_grid=True)
    episode = new_episode(push_test, np.random.default_rng(0))
    obs = observe_episode(episode, push_test)
    own, other = obs[:, :, :SIZE], obs[:, :, SIZE:]
    assert obs.shape == (len(CHANNEL_INDEX), SIZE, 2 * SIZE)  # the picture the network trained on
    assert channel(own, "body").sum() == 2 and channel(other, "body").sum() == 0
    assert channel(own, "health")[episode.state.npc.position] == 1.0  # the NPC's state is shown
    # The agent can push it: put the NPC right of the agent, with floor behind it, and walk into it
    state = episode.state
    state.tiles[3, 2:5], state.food[3, 2:5] = Tile.FLOOR, False
    state.agent.position, state.npc.position = (3, 2), (3, 3)
    _, result = step_episode(episode, Action.RIGHT, push_test, np.random.default_rng(0))
    assert result.npc_pushed and result.next_state.npc.position == (3, 4)


def test_the_push_test_needs_the_parallel_grid():
    try:
        GridworldConfig(npc_on_agent_grid=True)
    except ValueError:
        return
    raise AssertionError("expected a ValueError")


def test_batched_env_observations_have_both_grids():
    env = BatchedGridworldEnv(4, seed=0, device="cpu", config=PARALLEL)
    obs, _ = env.reset()
    assert env.single_observation_space.shape == (len(CHANNEL_INDEX), SIZE, 2 * SIZE)
    next_obs, *_ = env.step(t.zeros(4, dtype=t.long))
    assert obs.shape == next_obs.shape == (4, len(CHANNEL_INDEX), SIZE, 2 * SIZE)


def test_the_shared_grid_is_the_default():
    episode = new_episode(GridworldConfig(), np.random.default_rng(0))
    assert episode.npc_world is None
    assert observe_episode(episode, GridworldConfig()).shape == (len(CHANNEL_INDEX), SIZE, SIZE)
