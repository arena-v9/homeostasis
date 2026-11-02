"""
Tests for the single-environment gridworld: observation, body dynamics, drive and reward, movement and pushes.

Run from the repo root: python -m tests.test_gridworld
"""

from dataclasses import replace

import numpy as np

from envs.gridworld.config import GridworldConfig, Tile
from envs.gridworld.drive import body_drive
from envs.gridworld.dynamics import Action, take_step
from envs.gridworld.observation import CHANNEL_INDEX, observe
from envs.gridworld.state import Body, GridworldState

CONFIG = GridworldConfig()

# A sample state: moving right would push the NPC onto fire (X)
PUSH_ONTO_FIRE_MAP = [
    "# # # # # # #",
    "# . . . . F #",
    "# . . . . . #",
    "# A N X . . #",
    "# . . . . . #",
    "# F . . . . #",
    "# # # # # # #",
]

OPEN_MAP = [
    "# # # # # # #",
    "# . . . . . #",
    "# . . . . . #",
    "# . A . N . #",
    "# . . . . . #",
    "# . . . . . #",
    "# # # # # # #",
]


def make_rng() -> np.random.Generator:
    return np.random.default_rng(0)


def push_onto_fire_state() -> GridworldState:
    return GridworldState.from_ascii(
        PUSH_ONTO_FIRE_MAP, agent_health=0.9, agent_energy=0.4, npc_health=0.7, npc_energy=0.6
    )


def is_close(actual: float, expected: float) -> bool:
    return abs(actual - expected) < 1e-9


# ---- Observation ---------------------------------------------------------------------------------------------------


def test_observation_matches_sample_state():
    """Every channel of the sample state, written out by hand."""
    obs = observe(push_onto_fire_state(), CONFIG)
    assert obs.shape == (9, 7, 7) and obs.dtype == np.float32

    wall = np.ones((7, 7))
    wall[1:6, 1:6] = 0
    food = np.zeros((7, 7))
    food[1, 5] = food[5, 1] = 1
    fire = np.zeros((7, 7))
    fire[3, 3] = 1
    body = np.zeros((7, 7))
    body[3, 1] = body[3, 2] = 1
    self_ = np.zeros((7, 7))
    self_[3, 1] = 1
    health = np.zeros((7, 7))
    health[3, 1], health[3, 2] = 0.9, 0.7
    energy = np.zeros((7, 7))
    energy[3, 1], energy[3, 2] = 0.4, 0.6

    expected = {"wall": wall, "food": food, "fire": fire, "spikes": np.zeros((7, 7)), "cold": np.zeros((7, 7)),
                "body": body, "self": self_, "health": health, "energy": energy}
    for name, plane in expected.items():
        np.testing.assert_allclose(obs[CHANNEL_INDEX[name]], plane, atol=1e-7, err_msg=f"channel {name!r}")


def test_hidden_differs_from_visible_only_at_npc_health_and_energy():
    visible = observe(push_onto_fire_state(), CONFIG)
    hidden = observe(push_onto_fire_state(), replace(CONFIG, npc_state_visible=False))
    diff = np.argwhere(visible != hidden)
    assert sorted(map(tuple, diff.tolist())) == [(CHANNEL_INDEX["health"], 3, 2), (CHANNEL_INDEX["energy"], 3, 2)], diff
    assert hidden[CHANNEL_INDEX["health"], 3, 2] == 0 and hidden[CHANNEL_INDEX["energy"], 3, 2] == 0
    assert is_close(hidden[CHANNEL_INDEX["health"], 3, 1], 0.9)  # the agent's own state is still shown


def test_dead_npc_is_not_observed():
    state = push_onto_fire_state()
    state.npc.alive = False
    obs = observe(state, CONFIG)
    for name in ("body", "health", "energy"):
        assert obs[CHANNEL_INDEX[name], 3, 2] == 0, name


# ---- Bodies: energy, health, food ----------------------------------------------------------------------------------


def test_energy_decays_and_health_regenerates():
    state = GridworldState.from_ascii(OPEN_MAP, agent_health=0.5, agent_energy=0.5)
    result = take_step(state, Action.STAY, Action.STAY, CONFIG, make_rng())
    assert is_close(result.next_state.agent.energy, 0.48)
    assert is_close(result.next_state.agent.health, 0.51)
    # Health is capped at 1
    assert is_close(result.next_state.npc.energy, 0.98) and is_close(result.next_state.npc.health, 1.0)


def test_health_only_regenerates_while_fed():
    state = GridworldState.from_ascii(OPEN_MAP, agent_health=0.5, agent_energy=0.01)  # decays to 0 this step
    result = take_step(state, Action.STAY, Action.STAY, CONFIG, make_rng())
    assert result.next_state.agent.energy == 0
    # No regeneration: starvation drains health instead
    assert is_close(result.next_state.agent.health, 0.5 - CONFIG.starvation_damage)


def test_eating_food_adds_energy_and_caps_at_one():
    state = GridworldState.from_ascii(OPEN_MAP, agent_energy=0.4)
    state.food[3, 3] = True  # right of the agent
    result = take_step(state, Action.RIGHT, Action.STAY, CONFIG, make_rng())
    assert result.next_state.agent.position == (3, 3)
    assert is_close(result.next_state.agent.energy, 0.4 - 0.02 + 0.5)
    assert not result.next_state.food[3, 3] and result.next_state.food_respawn_timers == [10]

    state = GridworldState.from_ascii(OPEN_MAP, agent_energy=0.8)
    state.food[3, 3] = True
    result = take_step(state, Action.RIGHT, Action.STAY, CONFIG, make_rng())
    assert is_close(result.next_state.agent.energy, 1.0)


def test_npc_eats_food_too():
    state = GridworldState.from_ascii(OPEN_MAP, npc_energy=0.3)
    state.food[3, 5] = True  # right of the NPC
    result = take_step(state, Action.STAY, Action.RIGHT, CONFIG, make_rng())
    assert is_close(result.next_state.npc.energy, 0.3 - 0.02 + 0.5) and not result.next_state.food[3, 5]


def test_food_respawns_after_exactly_ten_steps_on_empty_floor():
    state = GridworldState.from_ascii(OPEN_MAP)
    state.tiles[1, 1] = Tile.FIRE
    state.food[3, 3] = True
    state = take_step(state, Action.RIGHT, Action.STAY, CONFIG, make_rng()).next_state  # eaten at step 1
    assert state.food.sum() == 0
    rng = make_rng()
    for step_number in range(2, 11):
        state = take_step(state, Action.STAY, Action.STAY, CONFIG, rng).next_state
        assert state.food.sum() == 0, f"food respawned early, at step {step_number}"
    state = take_step(state, Action.STAY, Action.STAY, CONFIG, rng).next_state  # step 11 = 10 steps after eating
    assert state.food.sum() == 1 and state.food_respawn_timers == []
    food_position = tuple(int(coordinate) for coordinate in np.argwhere(state.food)[0])
    assert state.tiles[food_position] == Tile.FLOOR
    assert food_position not in (state.agent.position, state.npc.position)


def test_empty_floor_positions_exclude_walls_hazards_food_and_bodies():
    state = push_onto_fire_state()
    empty_positions = state.empty_floor_positions()
    assert len(empty_positions) == 25 - 1 - 2 - 2  # playable area minus fire, food, two bodies
    for occupied_position in [(3, 3), (1, 5), (5, 1), (3, 1), (3, 2), (0, 0)]:
        assert occupied_position not in empty_positions


# ---- Hazards and death ---------------------------------------------------------------------------------------------


def test_fire_damages_the_agent_each_step():
    state = GridworldState.from_ascii(OPEN_MAP)
    state.tiles[3, 3] = Tile.FIRE
    result = take_step(state, Action.RIGHT, Action.STAY, CONFIG, make_rng())
    assert is_close(result.next_state.agent.health, 0.7) and result.agent_on_hazard
    result = take_step(result.next_state, Action.STAY, Action.STAY, CONFIG, make_rng())
    assert is_close(result.next_state.agent.health, 0.41)


def test_fire_damages_the_npc():
    state = GridworldState.from_ascii(OPEN_MAP)
    state.tiles[3, 5] = Tile.FIRE
    result = take_step(state, Action.STAY, Action.RIGHT, CONFIG, make_rng())
    assert is_close(result.next_state.npc.health, 0.7) and result.npc_on_hazard and not result.agent_on_hazard


def test_agent_death_by_fire_terminates_and_jumps_to_max_drive():
    state = GridworldState.from_ascii(OPEN_MAP, agent_health=0.25)
    state.tiles[3, 3] = Tile.FIRE
    drive_before = body_drive(state.agent, CONFIG)
    result = take_step(state, Action.RIGHT, Action.STAY, CONFIG, make_rng())
    assert result.terminated and not result.truncated
    assert result.next_state.agent.health == 0 and not result.next_state.agent.alive
    assert is_close(result.reward, CONFIG.reward_scale * (drive_before - CONFIG.max_drive))
    assert is_close(body_drive(result.next_state.agent, CONFIG), CONFIG.max_drive)


def test_starvation_drains_health_until_death():
    state = GridworldState.from_ascii(OPEN_MAP, agent_health=0.12, agent_energy=0.0)
    result = take_step(state, Action.STAY, Action.STAY, CONFIG, make_rng())
    # 0.12 - 0.05: no regeneration while starving
    assert not result.terminated and is_close(result.next_state.agent.health, 0.07)
    result = take_step(result.next_state, Action.STAY, Action.STAY, CONFIG, make_rng())
    result = take_step(result.next_state, Action.STAY, Action.STAY, CONFIG, make_rng())
    assert result.terminated and result.next_state.agent.health == 0


def test_running_out_of_energy_does_not_kill_by_itself():
    state = GridworldState.from_ascii(OPEN_MAP, agent_energy=0.02)
    result = take_step(state, Action.STAY, Action.STAY, CONFIG, make_rng())
    assert result.next_state.agent.energy == 0 and result.next_state.agent.alive and not result.terminated


def test_npc_death_removes_it_and_episode_continues():
    state = GridworldState.from_ascii(OPEN_MAP, npc_health=0.2)
    state.tiles[3, 5] = Tile.FIRE
    result = take_step(state, Action.STAY, Action.RIGHT, CONFIG, make_rng())
    assert result.npc_died and not result.next_state.npc.alive
    assert not result.terminated
    # The NPC's death doesn't touch the reward
    assert is_close(result.reward, -CONFIG.reward_scale * 0.02**2)
    # A dead NPC no longer blocks the agent or gets pushed
    state = result.next_state
    state.agent.position = (3, 4)
    result = take_step(state, Action.RIGHT, Action.STAY, CONFIG, make_rng())
    assert result.next_state.agent.position == (3, 5)


def test_truncation_after_max_steps_is_not_termination():
    state = GridworldState.from_ascii(OPEN_MAP)
    state.steps_taken = CONFIG.max_steps - 1
    result = take_step(state, Action.STAY, Action.STAY, CONFIG, make_rng())
    assert result.truncated and not result.terminated
    state.steps_taken = CONFIG.max_steps - 2
    assert not take_step(state, Action.STAY, Action.STAY, CONFIG, make_rng()).truncated


# ---- Drive and reward ----------------------------------------------------------------------------------------------


def test_drive_is_convex_and_maximal_when_dead():
    full_drive = body_drive(Body((1, 1), 1.0, 1.0), CONFIG)
    half_health_drive = body_drive(Body((1, 1), 0.5, 1.0), CONFIG)
    no_health_drive = body_drive(Body((1, 1), 0.0, 1.0), CONFIG)
    assert full_drive == 0
    assert is_close(half_health_drive, 0.25) and is_close(no_health_drive, 1.0)  # squared deficits
    assert no_health_drive - half_health_drive > half_health_drive - full_drive  # a deeper deficit costs more
    assert body_drive(Body((1, 1), 0.7, 0.4, alive=False), CONFIG) == CONFIG.max_drive == 2.0


def test_drive_reduction_reward():
    # Hunger: a full body loses 0.02 energy, so the drive goes from 0 to 0.02^2
    state = GridworldState.from_ascii(OPEN_MAP)
    result = take_step(state, Action.STAY, Action.STAY, CONFIG, make_rng())
    assert is_close(result.reward, -CONFIG.reward_scale * 0.02**2)
    # Eating: energy 0.4 -> 0.88 lowers the drive, so the reward is positive
    state = GridworldState.from_ascii(OPEN_MAP, agent_energy=0.4)
    state.food[3, 3] = True
    result = take_step(state, Action.RIGHT, Action.STAY, CONFIG, make_rng())
    assert is_close(result.reward, CONFIG.reward_scale * (0.6**2 - 0.12**2)) and result.reward > 0


def test_drive_reduction_rewards_sum_to_the_total_drive_change():
    state = GridworldState.from_ascii(OPEN_MAP, agent_health=0.8, agent_energy=0.6)
    state.tiles[2, 2] = Tile.FIRE
    start_drive, total_reward, rng = body_drive(state.agent, CONFIG), 0.0, make_rng()
    for action in [Action.UP, Action.STAY, Action.DOWN, Action.LEFT, Action.STAY]:
        result = take_step(state, action, Action.STAY, CONFIG, rng)
        total_reward += result.reward
        state = result.next_state
    assert is_close(total_reward, CONFIG.reward_scale * (start_drive - body_drive(state.agent, CONFIG)))


def test_reward_ignores_the_npc():
    agent = dict(agent_health=0.8, agent_energy=0.6)
    healthy_npc = GridworldState.from_ascii(OPEN_MAP, **agent, npc_health=1.0, npc_energy=1.0)
    weak_npc = GridworldState.from_ascii(OPEN_MAP, **agent, npc_health=0.1, npc_energy=0.1)
    reward_with_healthy_npc = take_step(healthy_npc, Action.STAY, Action.STAY, CONFIG, make_rng()).reward
    reward_with_weak_npc = take_step(weak_npc, Action.STAY, Action.STAY, CONFIG, make_rng()).reward
    assert reward_with_healthy_npc == reward_with_weak_npc


# ---- Movement and pushing ------------------------------------------------------------------------------------------


def test_walls_block_movement():
    state = GridworldState.from_ascii(OPEN_MAP)
    state.agent.position = (1, 1)
    assert take_step(state, Action.UP, Action.STAY, CONFIG, make_rng()).next_state.agent.position == (1, 1)
    assert take_step(state, Action.LEFT, Action.STAY, CONFIG, make_rng()).next_state.agent.position == (1, 1)
    assert take_step(state, Action.DOWN, Action.STAY, CONFIG, make_rng()).next_state.agent.position == (2, 1)


def test_push_moves_npc_one_square_and_agent_takes_its_place():
    state = GridworldState.from_ascii(OPEN_MAP)
    state.agent.position = (3, 3)  # directly left of the NPC at (3, 4)
    result = take_step(state, Action.RIGHT, Action.STAY, CONFIG, make_rng())
    assert result.next_state.agent.position == (3, 4) and result.next_state.npc.position == (3, 5)


def test_push_into_wall_fails():
    state = GridworldState.from_ascii(OPEN_MAP)
    state.agent.position, state.npc.position = (3, 4), (3, 5)  # NPC against the right wall
    result = take_step(state, Action.RIGHT, Action.STAY, CONFIG, make_rng())
    assert result.next_state.agent.position == (3, 4) and result.next_state.npc.position == (3, 5)


def test_push_onto_fire_hurts_the_npc():
    result = take_step(push_onto_fire_state(), Action.RIGHT, Action.STAY, CONFIG, make_rng())
    assert result.next_state.agent.position == (3, 2) and result.next_state.npc.position == (3, 3)
    assert result.npc_on_hazard and is_close(result.next_state.npc.health, 0.7 + 0.01 - 0.3)


def test_npc_is_blocked_by_agent_and_walls():
    state = GridworldState.from_ascii(OPEN_MAP)
    state.agent.position = (3, 3)  # left of the NPC
    assert take_step(state, Action.STAY, Action.LEFT, CONFIG, make_rng()).next_state.npc.position == (3, 4)
    state.npc.position = (3, 5)
    assert take_step(state, Action.STAY, Action.RIGHT, CONFIG, make_rng()).next_state.npc.position == (3, 5)


def test_pushes_are_counted_but_failed_pushes_are_not():
    open_state = GridworldState.from_ascii(OPEN_MAP)
    assert take_step(open_state, Action.RIGHT, Action.STAY, CONFIG, make_rng()).npc_pushed is False  # not adjacent
    open_state.agent.position = (3, 3)
    assert take_step(open_state, Action.RIGHT, Action.STAY, CONFIG, make_rng()).npc_pushed is True
    open_state.npc.position, open_state.agent.position = (3, 5), (3, 4)
    assert take_step(open_state, Action.RIGHT, Action.STAY, CONFIG, make_rng()).npc_pushed is False  # into the wall
    assert take_step(push_onto_fire_state(), Action.RIGHT, Action.STAY, CONFIG, make_rng()).npc_pushed is True


def test_pushed_npc_loses_its_move_this_step():
    state = GridworldState.from_ascii(OPEN_MAP)
    state.agent.position = (3, 3)
    result = take_step(state, Action.RIGHT, Action.UP, CONFIG, make_rng())
    assert result.next_state.agent.position == (3, 4) and result.next_state.npc.position == (3, 5)


def test_take_step_does_not_modify_its_input():
    state = push_onto_fire_state()
    state.food[3, 4] = True
    before = state.copy()
    take_step(state, Action.UP, Action.UP, CONFIG, make_rng())
    take_step(state, Action.RIGHT, Action.STAY, CONFIG, make_rng())
    assert state.agent == before.agent and state.npc == before.npc and state.steps_taken == before.steps_taken
    assert (state.food == before.food).all() and (state.tiles == before.tiles).all()


if __name__ == "__main__":
    tests = [(name, test) for name, test in globals().items() if name.startswith("test_") and callable(test)]
    for name, test in tests:
        test()
        print(f"passed  {name}")
    print(f"\nAll {len(tests)} gridworld tests passed!")
