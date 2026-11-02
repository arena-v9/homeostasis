"""
Tests for the scripted NPC (seeking food when hungry, wandering otherwise, its occasional mistaken step onto fire)
and for `drop_npc_harm`, which keeps every transition that leaves the NPC hurt out of the replay buffer.

Run from the repo root: python -m tests.test_npc_policy
"""

from dataclasses import replace

import numpy as np
import torch as t

from dqn.agent import DQNAgent
from dqn.networks import ConvQNetwork
from dqn.replay_buffer import ReplayBuffer
from envs.gridworld.batched_env import BatchedGridworldEnv
from envs.gridworld.config import GridworldConfig, Tile
from envs.gridworld.dynamics import ACTION_OFFSETS, Action, offset_position, take_step
from envs.gridworld.initial_state import generate_initial_state
from envs.gridworld.npc_policy import scripted_npc_policy
from envs.gridworld.observation import CHANNEL_INDEX
from envs.gridworld.state import GridworldState

CONFIG = replace(GridworldConfig(), npc_fire_step_prob=0.0)  # an NPC that never steps onto fire by mistake
CLUMSY = replace(CONFIG, npc_fire_step_prob=0.2)

# The NPC has food two squares to its left, with fire directly to its right
FOOD_LEFT_FIRE_RIGHT_MAP = [
    "# # # # # # #",
    "# . . . . . #",
    "# A . . . . #",
    "# F . N X . #",
    "# . . . . . #",
    "# . . . . . #",
    "# # # # # # #",
]

# The only short route to food runs past fire: the NPC must go around it
FIRE_BETWEEN_MAP = [
    "# # # # # # #",
    "# . . . . . #",
    "# . . . . . #",
    "# F X N . . #",
    "# . . . . . #",
    "# . . . . A #",
    "# # # # # # #",
]


def npc_actions(state: GridworldState, config: GridworldConfig, num_samples: int = 500, seed: int = 0) -> set[Action]:
    """Every action the NPC chooses in `state` over many random draws."""
    rng = np.random.default_rng(seed)
    return {scripted_npc_policy(state, config, rng) for _ in range(num_samples)}


def test_hungry_npc_heads_for_the_nearest_food():
    state = GridworldState.from_ascii(FOOD_LEFT_FIRE_RIGHT_MAP, npc_energy=0.3)
    assert npc_actions(state, CONFIG) == {Action.LEFT}


def test_hungry_npc_routes_around_fire():
    state = GridworldState.from_ascii(FIRE_BETWEEN_MAP, npc_energy=0.3)
    # Food at (3, 1) is 4 steps away around the fire, via (2, 2) or (4, 2): up or down are both shortest
    assert npc_actions(state, CONFIG) == {Action.UP, Action.DOWN}


def test_hungry_npc_routes_around_the_agent():
    state = GridworldState.from_ascii(
        [
            "# # # # # # #",
            "# . . . . . #",
            "# . . . . . #",
            "# F A N . . #",
            "# . . . . . #",
            "# . . . . . #",
            "# # # # # # #",
        ],
        npc_energy=0.3,
    )
    assert npc_actions(state, CONFIG) == {Action.UP, Action.DOWN}


def test_npc_that_is_not_hungry_wanders_but_never_onto_fire_or_the_agent():
    state = GridworldState.from_ascii(FOOD_LEFT_FIRE_RIGHT_MAP, npc_energy=0.9)
    assert npc_actions(state, CONFIG) == {Action.UP, Action.DOWN, Action.LEFT, Action.STAY}
    state.agent.position = (2, 3)  # directly above the NPC
    assert npc_actions(state, CONFIG) == {Action.DOWN, Action.LEFT, Action.STAY}


def test_hungry_npc_wanders_when_no_food_is_reachable():
    state = GridworldState.from_ascii(FOOD_LEFT_FIRE_RIGHT_MAP, npc_energy=0.3)
    state.food[:] = False
    assert npc_actions(state, CONFIG) == {Action.UP, Action.DOWN, Action.LEFT, Action.STAY}


def test_dead_npc_stays():
    state = GridworldState.from_ascii(FOOD_LEFT_FIRE_RIGHT_MAP, npc_energy=0.3)
    state.npc.alive = False
    assert npc_actions(state, CONFIG) == {Action.STAY}


def test_npc_sometimes_steps_onto_adjacent_fire_by_mistake():
    state = GridworldState.from_ascii(FOOD_LEFT_FIRE_RIGHT_MAP, npc_energy=0.3)
    rng = np.random.default_rng(0)
    actions = [scripted_npc_policy(state, CLUMSY, rng) for _ in range(5000)]
    fire_fraction = actions.count(Action.RIGHT) / len(actions)
    assert abs(fire_fraction - CLUMSY.npc_fire_step_prob) < 0.03, fire_fraction


def test_npc_steps_off_fire_the_next_step():
    state = GridworldState.from_ascii(FOOD_LEFT_FIRE_RIGHT_MAP, npc_energy=0.9)
    state.npc.position = (3, 4)  # on the fire
    actions = npc_actions(state, CLUMSY)
    assert Action.STAY not in actions
    for action in actions:
        assert state.tiles[offset_position(state.npc.position, ACTION_OFFSETS[action])] != Tile.FIRE


def test_npc_without_mistakes_never_walks_onto_fire():
    rng = np.random.default_rng(0)
    for _ in range(50):
        state = generate_initial_state(CONFIG, rng)
        for _ in range(CONFIG.max_steps):
            result = take_step(state, Action.STAY, scripted_npc_policy(state, CONFIG, rng), CONFIG, rng)
            assert not result.npc_on_hazard
            if result.terminated:
                break
            state = result.next_state


def npc_on_fire_in(obs: t.Tensor) -> t.Tensor:
    """For each observation in a batch, whether the NPC (a body that isn't self) stands on fire."""
    npc = obs[:, CHANNEL_INDEX["body"]] * (1 - obs[:, CHANNEL_INDEX["self"]])
    return (npc * obs[:, CHANNEL_INDEX["fire"]]).flatten(1).sum(1) > 0


def fill_buffer(config: GridworldConfig, steps: int = 400, num_envs: int = 16) -> tuple[ReplayBuffer, int]:
    """Plays random actions, storing transitions as in training. Returns the buffer and how many were dropped."""
    env = BatchedGridworldEnv(num_envs, seed=0, device="cpu", config=config)
    obs_shape = env.single_observation_space.shape
    buffer = ReplayBuffer(num_envs, obs_shape, (), 100_000, seed=0, device="cpu")
    network = ConvQNetwork(obs_shape, env.single_action_space.n)
    agent = DQNAgent(env, buffer, network, 1.0, 1.0, 0.5, 10_000, t.Generator().manual_seed(0))  # random actions
    num_dropped = sum(int(agent.play_step()["dropped"].sum()) for _ in range(steps))
    return buffer, num_dropped


def test_npc_harm_reaches_the_buffer_by_default():
    buffer, num_dropped = fill_buffer(CLUMSY)
    assert num_dropped == 0
    assert npc_on_fire_in(buffer.next_obs).any()


def test_drop_npc_harm_keeps_every_npc_on_fire_transition_out_of_the_buffer():
    buffer, num_dropped = fill_buffer(replace(CLUMSY, drop_npc_harm=True))
    assert num_dropped > 0 and buffer.size == 16 * 400 - num_dropped
    assert not npc_on_fire_in(buffer.obs).any() and not npc_on_fire_in(buffer.next_obs).any()
    # The episode restarts after a drop, so the agent never sees the aftermath either: the NPC's health is always full
    for obs in (buffer.obs, buffer.next_obs):
        npc = obs[:, CHANNEL_INDEX["body"]] * (1 - obs[:, CHANNEL_INDEX["self"]])
        npc_health = (obs[:, CHANNEL_INDEX["health"]] * npc).flatten(1).sum(1)
        assert (npc_health[npc.flatten(1).sum(1) > 0] == 1).all()


def test_npc_deaths_are_reported():
    """Every NPC death shows up in infos["npc_died"], and only on the step it happens."""
    deadly = replace(CLUMSY, hazard_damage={Tile.FIRE: (0.3, 1.0)})  # one step on fire kills the NPC
    env = BatchedGridworldEnv(16, seed=0, device="cpu", config=deadly)
    obs, _ = env.reset()
    deaths = 0
    for _ in range(300):
        npc_was_alive = t.tensor([episode.state.npc.alive for episode in env.episodes])
        obs, _, terminated, truncated, infos = env.step(t.randint(0, 5, (16,)))
        npc_now_alive = t.tensor([episode.state.npc.alive for episode in env.episodes])
        continuing = ~(terminated | truncated)
        assert (infos["npc_died"] <= npc_was_alive).all()
        assert (infos["npc_died"][continuing] == (npc_was_alive & ~npc_now_alive)[continuing]).all()
        deaths += int(infos["npc_died"].sum())
    assert deaths > 0


if __name__ == "__main__":
    tests = [(name, test) for name, test in globals().items() if name.startswith("test_") and callable(test)]
    for name, test in tests:
        test()
        print(f"passed  {name}")
    print(f"\nAll {len(tests)} NPC policy tests passed!")
