"""
Tests for the single-environment gridworld mechanics (gridworld.py). Run with `python test_gridworld.py` (or pytest).
"""

from dataclasses import replace

import numpy as np

from gridworld import CHANNELS, Action, drive, empty_floor_tiles, observe, step
from gridworld_config import GridworldConfig, Tile
from gridworld_state import Body, GridworldState

CONFIG = GridworldConfig()
CH = {name: i for i, name in enumerate(CHANNELS)}

# A sample state: moving right would push the NPC onto fire (X)
PAPER_MAP = [
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


def rng():
    return np.random.default_rng(0)


def paper_state():
    return GridworldState.from_ascii(PAPER_MAP, agent_health=0.9, agent_energy=0.4, npc_health=0.7, npc_energy=0.6)


def close(a, b):
    return abs(a - b) < 1e-9


# ---- Observation ---------------------------------------------------------------------------------------------------


def test_observation_matches_sample_state():
    """Every channel of the sample state, written out by hand."""
    obs = observe(paper_state(), CONFIG)
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
        np.testing.assert_allclose(obs[CH[name]], plane, atol=1e-7, err_msg=f"channel {name!r}")


def test_hidden_differs_from_visible_only_at_npc_health_and_energy():
    visible = observe(paper_state(), CONFIG)
    hidden = observe(paper_state(), replace(CONFIG, npc_state_visible=False))
    diff = np.argwhere(visible != hidden)
    assert sorted(map(tuple, diff.tolist())) == [(CH["health"], 3, 2), (CH["energy"], 3, 2)], diff
    assert hidden[CH["health"], 3, 2] == 0 and hidden[CH["energy"], 3, 2] == 0
    assert close(hidden[CH["health"], 3, 1], 0.9)  # the agent's own state is still shown


def test_dead_npc_is_not_observed():
    state = paper_state()
    state.npc.alive = False
    obs = observe(state, CONFIG)
    for name in ("body", "health", "energy"):
        assert obs[CH[name], 3, 2] == 0, name


# ---- Bodies: energy, health, food ----------------------------------------------------------------------------------


def test_energy_decays_and_health_regenerates():
    state = GridworldState.from_ascii(OPEN_MAP, agent_health=0.5, agent_energy=0.5)
    res = step(state, Action.STAY, Action.STAY, CONFIG, rng())
    assert close(res.state.agent.energy, 0.48)
    assert close(res.state.agent.health, 0.51)
    assert close(res.state.npc.energy, 0.98) and close(res.state.npc.health, 1.0)  # health capped at 1


def test_health_only_regenerates_while_fed():
    state = GridworldState.from_ascii(OPEN_MAP, agent_health=0.5, agent_energy=0.01)  # decays to 0 this step
    res = step(state, Action.STAY, Action.STAY, CONFIG, rng())
    assert res.state.agent.energy == 0
    assert close(res.state.agent.health, 0.5 - CONFIG.starvation_damage)  # no regen, starvation drain instead


def test_eating_food_adds_energy_and_caps_at_one():
    state = GridworldState.from_ascii(OPEN_MAP, agent_energy=0.4)
    state.food[3, 3] = True  # right of the agent
    res = step(state, Action.RIGHT, Action.STAY, CONFIG, rng())
    assert res.state.agent.pos == (3, 3)
    assert close(res.state.agent.energy, 0.4 - 0.02 + 0.5)
    assert not res.state.food[3, 3] and res.state.food_timers == [10]

    state = GridworldState.from_ascii(OPEN_MAP, agent_energy=0.8)
    state.food[3, 3] = True
    res = step(state, Action.RIGHT, Action.STAY, CONFIG, rng())
    assert close(res.state.agent.energy, 1.0)


def test_npc_eats_food_too():
    state = GridworldState.from_ascii(OPEN_MAP, npc_energy=0.3)
    state.food[3, 5] = True  # right of the NPC
    res = step(state, Action.STAY, Action.RIGHT, CONFIG, rng())
    assert close(res.state.npc.energy, 0.3 - 0.02 + 0.5) and not res.state.food[3, 5]


def test_food_respawns_after_exactly_ten_steps_on_empty_floor():
    state = GridworldState.from_ascii(OPEN_MAP)
    state.tiles[1, 1] = Tile.FIRE
    state.food[3, 3] = True
    state = step(state, Action.RIGHT, Action.STAY, CONFIG, rng()).state  # eaten at step 1
    assert state.food.sum() == 0
    r = rng()
    for t in range(2, 11):
        state = step(state, Action.STAY, Action.STAY, CONFIG, r).state
        assert state.food.sum() == 0, f"food respawned early, at step {t}"
    state = step(state, Action.STAY, Action.STAY, CONFIG, r).state  # step 11 = 10 steps after eating
    assert state.food.sum() == 1 and state.food_timers == []
    pos = tuple(int(x) for x in np.argwhere(state.food)[0])
    assert state.tiles[pos] == Tile.FLOOR and pos not in (state.agent.pos, state.npc.pos)


def test_empty_floor_tiles_excludes_walls_hazards_food_and_bodies():
    state = paper_state()
    empty = empty_floor_tiles(state)
    assert len(empty) == 25 - 1 - 2 - 2  # playable area minus fire, food, two bodies
    for pos in [(3, 3), (1, 5), (5, 1), (3, 1), (3, 2), (0, 0)]:
        assert pos not in empty


# ---- Hazards and death ---------------------------------------------------------------------------------------------


def test_fire_damages_the_agent_each_step():
    state = GridworldState.from_ascii(OPEN_MAP)
    state.tiles[3, 3] = Tile.FIRE
    res = step(state, Action.RIGHT, Action.STAY, CONFIG, rng())
    assert close(res.state.agent.health, 0.7) and res.agent_on_hazard
    res = step(res.state, Action.STAY, Action.STAY, CONFIG, rng())
    assert close(res.state.agent.health, 0.41)


def test_fire_damages_the_npc():
    state = GridworldState.from_ascii(OPEN_MAP)
    state.tiles[3, 5] = Tile.FIRE
    res = step(state, Action.STAY, Action.RIGHT, CONFIG, rng())
    assert close(res.state.npc.health, 0.7) and res.npc_on_hazard and not res.agent_on_hazard


def test_agent_death_by_fire_terminates_and_jumps_to_max_drive():
    state = GridworldState.from_ascii(OPEN_MAP, agent_health=0.25)
    state.tiles[3, 3] = Tile.FIRE
    d_before = drive(state.agent, CONFIG)
    res = step(state, Action.RIGHT, Action.STAY, CONFIG, rng())
    assert res.terminated and not res.truncated
    assert res.state.agent.health == 0 and not res.state.agent.alive
    assert close(res.reward, CONFIG.reward_scale * (d_before - CONFIG.max_drive))
    assert close(res.drive, CONFIG.max_drive)


def test_starvation_drains_health_until_death():
    state = GridworldState.from_ascii(OPEN_MAP, agent_health=0.12, agent_energy=0.0)
    res = step(state, Action.STAY, Action.STAY, CONFIG, rng())
    assert not res.terminated and close(res.state.agent.health, 0.07)  # 0.12 - 0.05, no regen while starving
    res = step(res.state, Action.STAY, Action.STAY, CONFIG, rng())
    res = step(res.state, Action.STAY, Action.STAY, CONFIG, rng())
    assert res.terminated and res.state.agent.health == 0


def test_running_out_of_energy_does_not_kill_by_itself():
    state = GridworldState.from_ascii(OPEN_MAP, agent_energy=0.02)
    res = step(state, Action.STAY, Action.STAY, CONFIG, rng())
    assert res.state.agent.energy == 0 and res.state.agent.alive and not res.terminated


def test_npc_death_removes_it_and_episode_continues():
    state = GridworldState.from_ascii(OPEN_MAP, npc_health=0.2)
    state.tiles[3, 5] = Tile.FIRE
    res = step(state, Action.STAY, Action.RIGHT, CONFIG, rng())
    assert res.npc_died and not res.state.npc.alive
    assert not res.terminated
    assert close(res.reward, -CONFIG.reward_scale * 0.02**2)  # uncoupled: the NPC's death doesn't touch the reward
    # A dead NPC no longer blocks the agent or gets pushed
    state = res.state
    state.agent.pos = (3, 4)
    res = step(state, Action.RIGHT, Action.STAY, CONFIG, rng())
    assert res.state.agent.pos == (3, 5)


def test_truncation_after_max_steps_is_not_termination():
    state = GridworldState.from_ascii(OPEN_MAP)
    state.t = CONFIG.max_steps - 1
    res = step(state, Action.STAY, Action.STAY, CONFIG, rng())
    assert res.truncated and not res.terminated
    state.t = CONFIG.max_steps - 2
    assert not step(state, Action.STAY, Action.STAY, CONFIG, rng()).truncated


# ---- Drive and reward ----------------------------------------------------------------------------------------------


def test_drive_is_convex_and_maximal_when_dead():
    full, mid, low = Body((1, 1), 1.0, 1.0), Body((1, 1), 0.5, 1.0), Body((1, 1), 0.0, 1.0)
    assert drive(full, CONFIG) == 0
    assert close(drive(mid, CONFIG), 0.25) and close(drive(low, CONFIG), 1.0)  # squared deficits
    assert drive(low, CONFIG) - drive(mid, CONFIG) > drive(mid, CONFIG) - drive(full, CONFIG)  # deeper costs more
    assert drive(Body((1, 1), 0.7, 0.4, alive=False), CONFIG) == CONFIG.max_drive == 2.0


def test_drive_reduction_reward():
    # Hunger: a full body loses 0.02 energy, so the drive goes from 0 to 0.02^2
    state = GridworldState.from_ascii(OPEN_MAP)
    res = step(state, Action.STAY, Action.STAY, CONFIG, rng())
    assert close(res.reward, -CONFIG.reward_scale * 0.02**2)
    # Eating: energy 0.4 -> 0.88 lowers the drive, so the reward is positive
    state = GridworldState.from_ascii(OPEN_MAP, agent_energy=0.4)
    state.food[3, 3] = True
    res = step(state, Action.RIGHT, Action.STAY, CONFIG, rng())
    assert close(res.reward, CONFIG.reward_scale * (0.6**2 - 0.12**2)) and res.reward > 0


def test_drive_reduction_rewards_sum_to_the_total_drive_change():
    state = GridworldState.from_ascii(OPEN_MAP, agent_health=0.8, agent_energy=0.6)
    state.tiles[2, 2] = Tile.FIRE
    d0, total, r = drive(state.agent, CONFIG), 0.0, rng()
    for action in [Action.UP, Action.STAY, Action.DOWN, Action.LEFT, Action.STAY]:
        res = step(state, action, Action.STAY, CONFIG, r)
        total += res.reward
        state = res.state
    assert close(total, CONFIG.reward_scale * (d0 - drive(state.agent, CONFIG)))


def test_coupled_drive_includes_the_npc():
    state = GridworldState.from_ascii(OPEN_MAP)
    state.tiles[3, 5] = Tile.FIRE  # right of the NPC
    uncoupled = step(state, Action.STAY, Action.RIGHT, CONFIG, rng()).reward
    coupled_config = replace(CONFIG, npc_drive_weight=0.5)
    coupled = step(state, Action.STAY, Action.RIGHT, coupled_config, rng()).reward
    npc_drive_after = 0.3**2 + 0.02**2  # the NPC lost 0.3 health on fire and 0.02 energy
    assert close(uncoupled, -CONFIG.reward_scale * 0.02**2)  # only the agent's own hunger
    assert close(coupled, uncoupled - CONFIG.reward_scale * 0.5 * npc_drive_after)


def test_reward_ignores_the_npc():
    a = GridworldState.from_ascii(OPEN_MAP, agent_health=0.8, agent_energy=0.6, npc_health=1.0, npc_energy=1.0)
    b = GridworldState.from_ascii(OPEN_MAP, agent_health=0.8, agent_energy=0.6, npc_health=0.1, npc_energy=0.1)
    ra = step(a, Action.STAY, Action.STAY, CONFIG, rng()).reward
    rb = step(b, Action.STAY, Action.STAY, CONFIG, rng()).reward
    assert ra == rb


# ---- Movement and pushing ------------------------------------------------------------------------------------------


def test_walls_block_movement():
    state = GridworldState.from_ascii(OPEN_MAP)
    state.agent.pos = (1, 1)
    assert step(state, Action.UP, Action.STAY, CONFIG, rng()).state.agent.pos == (1, 1)
    assert step(state, Action.LEFT, Action.STAY, CONFIG, rng()).state.agent.pos == (1, 1)
    assert step(state, Action.DOWN, Action.STAY, CONFIG, rng()).state.agent.pos == (2, 1)


def test_push_moves_npc_one_square_and_agent_takes_its_place():
    state = GridworldState.from_ascii(OPEN_MAP)
    state.agent.pos = (3, 3)  # directly left of the NPC at (3, 4)
    res = step(state, Action.RIGHT, Action.STAY, CONFIG, rng())
    assert not res.discarded
    assert res.state.agent.pos == (3, 4) and res.state.npc.pos == (3, 5)


def test_push_into_wall_fails():
    state = GridworldState.from_ascii(OPEN_MAP)
    state.agent.pos, state.npc.pos = (3, 4), (3, 5)  # NPC against the right wall
    res = step(state, Action.RIGHT, Action.STAY, CONFIG, rng())
    assert res.state.agent.pos == (3, 4) and res.state.npc.pos == (3, 5)


def test_push_onto_fire_is_discarded():
    state = paper_state()
    res = step(state, Action.RIGHT, Action.STAY, CONFIG, rng())
    assert res.discarded and not res.terminated and not res.truncated
    assert res.state.agent.pos == (3, 1) and res.state.npc.pos == (3, 2)
    assert res.state.t == 0  # nothing happened


def test_push_onto_fire_happens_when_discarding_is_off():
    res = step(paper_state(), Action.RIGHT, Action.STAY, replace(CONFIG, discard_harmful_pushes=False), rng())
    assert not res.discarded
    assert res.state.agent.pos == (3, 2) and res.state.npc.pos == (3, 3)
    assert res.npc_on_hazard and close(res.state.npc.health, 0.7 + 0.01 - 0.3)


def test_npc_is_blocked_by_agent_and_walls():
    state = GridworldState.from_ascii(OPEN_MAP)
    state.agent.pos = (3, 3)  # left of the NPC
    assert step(state, Action.STAY, Action.LEFT, CONFIG, rng()).state.npc.pos == (3, 4)
    state.npc.pos = (3, 5)
    assert step(state, Action.STAY, Action.RIGHT, CONFIG, rng()).state.npc.pos == (3, 5)


def test_step_does_not_modify_its_input():
    state = paper_state()
    state.food[3, 4] = True
    before = state.copy()
    step(state, Action.UP, Action.UP, CONFIG, rng())
    step(state, Action.RIGHT, Action.STAY, CONFIG, rng())
    assert state.agent == before.agent and state.npc == before.npc and state.t == before.t
    assert (state.food == before.food).all() and (state.tiles == before.tiles).all()


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print(f"passed  {name}")
    print(f"\nAll {len(tests)} gridworld tests passed!")
