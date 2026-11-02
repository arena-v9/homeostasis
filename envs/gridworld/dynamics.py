"""
How one gridworld episode moves forward: `take_step` applies both bodies' actions to a state and returns the next
state and the agent's reward.

`take_step` takes the NPC's action as an argument instead of choosing it, so the dynamics can be tested on hand-built
states. During training the NPC's policy (npc_policy.py) supplies it.

Order of one step:
  1. The agent moves. Moving into the NPC pushes it one square and the agent takes its old square; a push into a
     wall fails and neither body moves.
  2. The NPC moves, unless the agent just pushed it: its action was chosen from the state before the step, so applying
     it from the square it was pushed to could walk it onto fire. Walls and the agent block it; it never pushes.
  3. Eaten food items count down, and any that reach 0 respawn on a random empty floor square.
  4. Each living body updates (see `_update_body`): energy falls by `energy_decay` and rises by eating; health
     regenerates while energy > 0, takes the damage of the tile it stands on, and drains by `starvation_damage`
     while energy is 0.
  5. A body dies when its health reaches 0. The agent's death terminates the episode; the NPC's death removes it from
     the map. The reward is the reduction in the agent's drive (see drive.py), and the episode is truncated after
     `max_steps` steps.
"""

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from envs.gridworld.config import GridworldConfig, Tile
from envs.gridworld.drive import drive_reduction_reward
from envs.gridworld.state import Body, GridworldState, Position


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4


ACTION_OFFSETS: dict[Action, Position] = {
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
    Action.STAY: (0, 0),
}


@dataclass
class StepResult:
    next_state: GridworldState
    reward: float
    terminated: bool  # the agent died
    truncated: bool  # the episode hit max_steps (not a terminal state: bootstrap through it)
    agent_on_hazard: bool = False
    npc_on_hazard: bool = False
    npc_pushed: bool = False  # the agent pushed the NPC this step
    npc_died: bool = False


def offset_position(position: Position, offset: Position) -> Position:
    return (position[0] + offset[0], position[1] + offset[1])


def _move_agent(state: GridworldState, action: Action) -> None:
    """Applies the agent's move (and any push) to `state` in place."""
    offset = ACTION_OFFSETS[Action(action)]
    target = offset_position(state.agent.position, offset)
    if state.tiles[target] == Tile.WALL:
        return

    if state.npc.alive and target == state.npc.position and offset != (0, 0):
        push_target = offset_position(state.npc.position, offset)
        if state.tiles[push_target] == Tile.WALL:
            return  # push fails: neither body moves
        state.npc.position = push_target

    state.agent.position = target


def _move_npc(state: GridworldState, action: Action) -> None:
    """Applies the NPC's move to `state` in place. Walls and the agent block it."""
    if not state.npc.alive:
        return
    target = offset_position(state.npc.position, ACTION_OFFSETS[Action(action)])
    if state.tiles[target] == Tile.WALL or target == state.agent.position:
        return
    state.npc.position = target


def _respawn_food(state: GridworldState, rng: np.random.Generator) -> None:
    """Counts down each eaten item's timer and puts it back on a random empty floor square when it reaches 0."""
    remaining_timers = []
    for timer in state.food_respawn_timers:
        timer -= 1
        if timer <= 0:
            empty_positions = state.empty_floor_positions()
            if empty_positions:
                state.food[empty_positions[rng.integers(len(empty_positions))]] = True
                continue
            timer = 0  # nowhere to respawn: try again next step
        remaining_timers.append(timer)
    state.food_respawn_timers = remaining_timers


def _update_body(state: GridworldState, body: Body, damage: float, config: GridworldConfig) -> None:
    """Updates one living body's energy and health for this step, in place (see step 4 in the module docstring)."""
    # Energy: a resource that runs down every step and is restored by eating
    body.energy -= config.energy_decay
    if state.food[body.position]:
        state.food[body.position] = False
        state.food_respawn_timers.append(config.food_respawn_steps)
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


def _update_npc(state: GridworldState, config: GridworldConfig) -> tuple[bool, bool]:
    """Updates a living NPC's body in place. Returns whether it's on a harmful tile and whether it died this step."""
    if not state.npc.alive:
        return False, False
    damage = config.npc_damage(state.tiles[state.npc.position])
    _update_body(state, state.npc, damage, config)
    return damage > 0, not state.npc.alive


def step_npc_world(
    world: GridworldState, npc_action: Action, config: GridworldConfig, rng: np.random.Generator
) -> tuple[GridworldState, bool, bool]:
    """
    One step of the NPC's own grid, in the parallel-grid setting (see episode.py): the same rules as `take_step`, with
    no agent. Returns the next state, whether the NPC is on a harmful tile and whether it died this step. Doesn't
    modify `world`.
    """
    next_world = world.copy()
    _move_npc(next_world, npc_action)
    _respawn_food(next_world, rng)
    on_hazard, died = _update_npc(next_world, config)
    next_world.steps_taken += 1
    return next_world, on_hazard, died


def take_step(
    state: GridworldState,
    agent_action: Action,
    npc_action: Action,
    config: GridworldConfig,
    rng: np.random.Generator,
) -> StepResult:
    """
    Advances the episode by one step. Doesn't modify `state`; the new state is `StepResult.next_state`.
    """
    next_state = state.copy()

    # 1. Agent moves (possibly pushing the NPC)
    _move_agent(next_state, agent_action)

    # 2. NPC moves, unless it was just pushed (its action was chosen from the square it was pushed off)
    npc_pushed = next_state.npc.position != state.npc.position
    if not npc_pushed:
        _move_npc(next_state, npc_action)

    # 3. Food respawns (before eating, so an item eaten at step t reappears at step t + food_respawn_steps)
    _respawn_food(next_state, rng)

    # 4. Bodies: energy, health, hazard damage, eating
    agent_tile = next_state.tiles[next_state.agent.position]
    _update_body(next_state, next_state.agent, config.agent_damage(agent_tile), config)
    npc_on_hazard, npc_died = _update_npc(next_state, config)

    # 5. Reward, termination, truncation
    next_state.steps_taken += 1
    terminated = not next_state.agent.alive
    truncated = not terminated and next_state.steps_taken >= config.max_steps

    return StepResult(
        next_state=next_state,
        reward=drive_reduction_reward(state, next_state, config),
        terminated=terminated,
        truncated=truncated,
        agent_on_hazard=config.agent_damage(agent_tile) > 0,
        npc_on_hazard=npc_on_hazard,
        npc_pushed=npc_pushed,
        npc_died=npc_died,
    )
