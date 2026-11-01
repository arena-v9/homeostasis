"""
Single-environment gridworld mechanics: `step` (one transition) and `observe` (the agent's 9-channel view).

`step` takes the NPC's action as an argument instead of choosing it, so the mechanics can be tested on hand-built
states. The scripted NPC (npc_policy.py) and the batched env wrapper supply it during training.

Order of one step:
  1. The agent moves. Moving into the NPC pushes it one square and the agent takes its old square; a push into a
     wall fails and neither body moves. If the push would put the NPC on a tile that hurts it (and
     `config.discard_harmful_pushes`), nothing happens and the result is flagged `discarded`, so the caller can
     drop the transition and restart the episode.
  2. The NPC moves. Walls and the agent block it; it never pushes.
  3. Eaten food items count down, and any that reach 0 respawn on a random empty floor tile.
  4. Each living body updates (see `_update_body`): energy falls by `energy_decay` and rises by eating; health
     regenerates while energy > 0, takes the damage of the tile it stands on, and drains by `starvation_damage`
     while energy is 0.
  5. A body dies when its health reaches 0. The agent's death terminates the episode; the NPC's death removes it from
     the map. The reward is the reduction in the agent's drive (see `drive` and `_reward`), and the episode is
     truncated after `max_steps` steps.
"""

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from envs.gridworld.config import GridworldConfig, Tile
from envs.gridworld.state import Body, GridworldState, Pos


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4


ACTION_DELTAS: dict[Action, Pos] = {
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
    Action.STAY: (0, 0),
}

# Observation channels, in order. Each is a (size, size) plane.
CHANNELS = ("wall", "food", "fire", "spikes", "cold", "body", "self", "health", "energy")
NUM_CHANNELS = len(CHANNELS)
_TILE_CHANNELS = {"wall": Tile.WALL, "fire": Tile.FIRE, "spikes": Tile.SPIKES, "cold": Tile.COLD}


@dataclass
class StepResult:
    state: GridworldState
    reward: float
    terminated: bool  # the agent died
    truncated: bool  # the episode hit max_steps (not a terminal state: bootstrap through it)
    discarded: bool  # the agent tried a harmful push; state is unchanged and the transition should be dropped
    drive: float = 0.0  # the agent's (possibly coupled) drive after the step
    agent_on_hazard: bool = False  # for the verification logs
    npc_on_hazard: bool = False
    npc_died: bool = False


def _add(pos: Pos, delta: Pos) -> Pos:
    return (pos[0] + delta[0], pos[1] + delta[1])


def _move_agent(state: GridworldState, action: Action, config: GridworldConfig) -> bool:
    """Applies the agent's move (and any push) to `state` in place. Returns True if the push was harmful and
    should be discarded, in which case `state` is left unchanged."""
    delta = ACTION_DELTAS[Action(action)]
    target = _add(state.agent.pos, delta)
    if state.tiles[target] == Tile.WALL:
        return False

    if state.npc.alive and target == state.npc.pos and delta != (0, 0):
        push_target = _add(state.npc.pos, delta)
        if state.tiles[push_target] == Tile.WALL:
            return False  # push fails: neither body moves
        if config.discard_harmful_pushes and config.npc_damage(state.tiles[push_target]) > 0:
            return True
        state.npc.pos = push_target

    state.agent.pos = target
    return False


def _move_npc(state: GridworldState, action: Action) -> None:
    if not state.npc.alive:
        return
    target = _add(state.npc.pos, ACTION_DELTAS[Action(action)])
    if state.tiles[target] == Tile.WALL or target == state.agent.pos:
        return
    state.npc.pos = target


def empty_floor_tiles(state: GridworldState) -> list[Pos]:
    """Floor tiles with no food and no (living) body on them."""
    occupied = {state.agent.pos} | ({state.npc.pos} if state.npc.alive else set())
    rows, cols = np.nonzero((state.tiles == Tile.FLOOR) & ~state.food)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if (int(r), int(c)) not in occupied]


def _respawn_food(state: GridworldState, rng: np.random.Generator) -> None:
    timers = []
    for timer in state.food_timers:
        timer -= 1
        if timer <= 0:
            empty = empty_floor_tiles(state)
            if empty:
                state.food[empty[rng.integers(len(empty))]] = True
                continue
            timer = 0  # nowhere to respawn: try again next step
        timers.append(timer)
    state.food_timers = timers


def _update_body(state: GridworldState, body: Body, damage: float, config: GridworldConfig) -> None:
    # Energy: a resource that runs down every step and is restored by eating
    body.energy -= config.energy_decay
    if state.food[body.pos]:
        state.food[body.pos] = False
        state.food_timers.append(config.food_respawn_steps)
        body.energy = min(1.0, body.energy + config.food_energy)
    body.energy = max(0.0, body.energy)

    # Health: bodily integrity. Regenerates while fed, is hurt by hazards, and drains while starving.
    if body.energy > 0:
        body.health = min(1.0, body.health + config.health_regen)
    else:
        body.health -= config.starvation_damage
    body.health = max(0.0, body.health - damage)
    if body.health <= 0:
        body.alive = False


def drive(body: Body, config: GridworldConfig) -> float:
    """How far a body is from its setpoints (health = energy = 1): (dh^n + de^n)^(1/m) for deficits dh, de. A dead
    body has the maximum drive."""
    if not body.alive:
        return config.max_drive
    n, m = config.drive_exponent, config.drive_root
    return ((1.0 - body.health) ** n + (1.0 - body.energy) ** n) ** (1.0 / m)


def total_drive(state: GridworldState, config: GridworldConfig) -> float:
    """The drive the agent's reward is computed from: its own, plus `npc_drive_weight` times the NPC's (a dead or
    absent NPC counts as maximum drive, but only matters if the weight is non-zero)."""
    d = drive(state.agent, config)
    if config.npc_drive_weight:
        d += config.npc_drive_weight * drive(state.npc, config)
    return d


def _reward(before: GridworldState, after: GridworldState, config: GridworldConfig) -> float:
    """Drive reduction: positive when the step brought the agent closer to its setpoints, negative when it moved away.
    A step that kills the agent jumps its drive to the maximum."""
    return config.reward_scale * (total_drive(before, config) - total_drive(after, config))


def step(
    state: GridworldState,
    agent_action: Action,
    npc_action: Action,
    config: GridworldConfig,
    rng: np.random.Generator,
) -> StepResult:
    """Advances one step. Doesn't modify `state`; the new state is in the result."""
    new = state.copy()

    # 1. Agent moves (possibly pushing the NPC)
    if _move_agent(new, agent_action, config):
        return StepResult(state=state.copy(), reward=0.0, terminated=False, truncated=False, discarded=True)

    # 2. NPC moves
    _move_npc(new, npc_action)

    # 3. Food respawns (before eating, so an item eaten at step t reappears at step t + food_respawn_steps)
    _respawn_food(new, rng)

    # 4. Bodies: energy, health, hazard damage, eating
    agent_tile = new.tiles[new.agent.pos]
    _update_body(new, new.agent, config.agent_damage(agent_tile), config)
    npc_on_hazard = npc_died = False
    if new.npc.alive:
        npc_tile = new.tiles[new.npc.pos]
        npc_on_hazard = config.npc_damage(npc_tile) > 0
        _update_body(new, new.npc, config.npc_damage(npc_tile), config)
        npc_died = not new.npc.alive

    # 5. Reward, termination, truncation
    new.t += 1
    terminated = not new.agent.alive
    reward = _reward(state, new, config)
    truncated = not terminated and new.t >= config.max_steps

    return StepResult(
        state=new,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        discarded=False,
        drive=total_drive(new, config),
        agent_on_hazard=config.agent_damage(agent_tile) > 0,
        npc_on_hazard=npc_on_hazard,
        npc_died=npc_died,
    )


def observe(state: GridworldState, config: GridworldConfig) -> np.ndarray:
    """
    The agent's view of the whole map: a (NUM_CHANNELS, size, size) float32 array with channels CHANNELS.

    Both bodies appear on the same `body`, `health` and `energy` channels - that shared representation is what the
    experiment tests. Only `self` tells them apart. In the hidden condition (`npc_state_visible=False`), `health`
    and `energy` are 0 at the NPC's square. A dead NPC is not shown at all.
    """
    obs = np.zeros((NUM_CHANNELS, state.size, state.size), dtype=np.float32)
    ch = {name: i for i, name in enumerate(CHANNELS)}

    for name, tile in _TILE_CHANNELS.items():
        obs[ch[name]] = state.tiles == tile
    obs[ch["food"]] = state.food

    bodies = [state.agent] + ([state.npc] if state.npc.alive else [])
    for body in bodies:
        obs[ch["body"]][body.pos] = 1.0
    obs[ch["self"]][state.agent.pos] = 1.0

    obs[ch["health"]][state.agent.pos] = state.agent.health
    obs[ch["energy"]][state.agent.pos] = state.agent.energy
    if state.npc.alive and config.npc_state_visible:
        obs[ch["health"]][state.npc.pos] = state.npc.health
        obs[ch["energy"]][state.npc.pos] = state.npc.energy

    return obs
