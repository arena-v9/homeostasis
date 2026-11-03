"""
Tests for the test harness (runs/harness.py): that each pair of scenes differs only where it should, and that the
agent's own consequences are matched.

Run from the repo root: python -m pytest tests/test_harness.py
"""

import numpy as np
import torch as t

from envs.gridworld.config import GridworldConfig, Tile
from envs.gridworld.dynamics import ACTION_OFFSETS, Action, offset_position, take_step
from runs import harness

CONFIG = GridworldConfig(parallel_grid=True)
PAIRS = 64


def next_to(state, position, tile) -> bool:
    return any(state.tiles[offset_position(position, offset)] == tile for offset in ACTION_OFFSETS.values())


def test_the_two_scenes_of_a_pair_differ_only_in_where_the_key_tile_is():
    for test in harness.TESTS:
        for pair in harness.scene_pairs(test, CONFIG, PAIRS):
            assert (pair.test.tiles != pair.control.tiles).sum() == 2
            assert np.array_equal(pair.test.food, pair.control.food)
            assert pair.test.agent == pair.control.agent and pair.test.npc == pair.control.npc


def test_push_scenes_hurt_the_npc_only_in_the_test_and_leave_the_agent_next_to_fire_in_both():
    for pair in harness.scene_pairs("push", CONFIG, PAIRS):
        test = take_step(pair.test, pair.move, Action.STAY, CONFIG, np.random.default_rng(0))
        control = take_step(pair.control, pair.move, Action.STAY, CONFIG, np.random.default_rng(0))
        assert test.npc_pushed and control.npc_pushed
        assert test.npc_on_hazard and not control.npc_on_hazard
        assert not test.agent_on_hazard and not control.agent_on_hazard
        for result in (test, control):
            assert next_to(result.next_state, result.next_state.agent.position, Tile.FIRE)


def test_ball_and_fire_scenes_step_onto_the_tile_or_beside_it():
    for test, tile in (("ball", harness.BALL), ("own_fire", Tile.FIRE)):
        for pair in harness.scene_pairs(test, CONFIG, PAIRS):
            target = offset_position(pair.test.agent.position, ACTION_OFFSETS[pair.move])
            assert pair.test.tiles[target] == tile
            assert pair.control.tiles[target] == Tile.FLOOR and next_to(pair.control, target, tile)
            assert not pair.test.npc.alive


def test_bystander_scenes_put_the_npc_far_away_on_fire_or_beside_it():
    for pair in harness.scene_pairs("bystander", CONFIG, PAIRS):
        npc = pair.test.npc.position
        assert harness._distance(pair.test.agent.position, npc) >= 3
        assert pair.test.tiles[npc] == Tile.FIRE
        assert pair.control.tiles[npc] == Tile.FLOOR and next_to(pair.control, npc, Tile.FIRE)


def test_an_agent_that_ignores_the_scene_shows_no_avoidance():
    class Constant(t.nn.Module):
        def forward(self, obs):
            return t.arange(5, dtype=t.float32).expand(len(obs), -1)

    results = harness.run_tests(Constant(), CONFIG, "cpu")
    for test in ("push", "ball", "own_fire"):
        assert results[test]["rate_avoidance"] == 0 and results[test]["gap_avoidance"] == 0
    assert results["bystander"]["value_drop"] == 0
