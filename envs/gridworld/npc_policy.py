"""
How the NPC chooses its action. The NPC has no network: it follows a fixed script (`scripted_npc_policy`):

  - Hungry (energy below `npc_hunger_threshold`): takes a step along a shortest path to the nearest food, never
    entering a tile that hurts it. Ties between equally short paths are broken at random.
  - Otherwise: wanders, choosing uniformly among staying and the moves onto squares that don't hurt it.
  - Mistakes: when next to fire (and not already on it), it first steps onto a random adjacent fire tile with
    probability `npc_fire_step_prob`. On the next step its normal script takes it off again, since it never chooses to
    enter or stay on a harmful tile. With `npc_fire_step_prob` = 0 it never stands on fire unless pushed there.

The agent's square counts as blocked, so a hungry NPC routes around the agent when it can. When the agent blocks every
shortest path, the NPC steps toward food as if the agent weren't there (and is blocked this step); if no food can be
reached safely at all, it wanders.
"""

from collections import deque
from typing import Callable

import numpy as np

from envs.gridworld.config import GridworldConfig, Tile
from envs.gridworld.dynamics import ACTION_OFFSETS, Action, offset_position
from envs.gridworld.state import GridworldState, Position

# Chooses the NPC's action from the current state
NpcPolicy = Callable[[GridworldState, GridworldConfig, np.random.Generator], Action]

MOVES = [action for action in Action if action != Action.STAY]


def stationary_npc_policy(state: GridworldState, config: GridworldConfig, rng: np.random.Generator) -> Action:
    """The NPC never moves."""
    return Action.STAY


def _is_safe(state: GridworldState, position: Position, config: GridworldConfig) -> bool:
    """Whether the NPC can stand on `position` without being hurt (walls don't count: it can't stand there)."""
    tile = state.tiles[position]
    return tile != Tile.WALL and config.npc_damage(tile) == 0


def _neighbour(state: GridworldState, action: Action) -> Position:
    return offset_position(state.npc.position, ACTION_OFFSETS[action])


def _distances_to_food(state: GridworldState, config: GridworldConfig, blocked: set[Position]) -> dict[Position, int]:
    """
    Steps from each square to the nearest food, moving only through safe squares not in `blocked` (breadth-first
    search outward from every food item at once). Squares that can't reach food are missing.
    """
    food_positions = [(int(row), int(col)) for row, col in np.argwhere(state.food)]
    distances = {position: 0 for position in food_positions if position not in blocked}
    queue = deque(distances)
    while queue:
        position = queue.popleft()
        for offset in ACTION_OFFSETS.values():
            neighbour = offset_position(position, offset)
            if neighbour not in distances and neighbour not in blocked and _is_safe(state, neighbour, config):
                distances[neighbour] = distances[position] + 1
                queue.append(neighbour)
    return distances


def _step_toward_food(state: GridworldState, config: GridworldConfig, rng: np.random.Generator) -> Action | None:
    """
    A random first move along a shortest safe path to the nearest food, or None if no food can be reached safely.
    Paths avoid the agent's square if possible.
    """
    for blocked in ({state.agent.position}, set()):
        distances = _distances_to_food(state, config, blocked)
        move_distances = {
            action: distances[_neighbour(state, action)] for action in MOVES if _neighbour(state, action) in distances
        }
        if move_distances:
            nearest = min(move_distances.values())
            best_moves = [action for action, distance in move_distances.items() if distance == nearest]
            return best_moves[rng.integers(len(best_moves))]
    return None


def _wander(state: GridworldState, config: GridworldConfig, rng: np.random.Generator) -> Action:
    """A uniformly random choice among staying (if the NPC's square is safe) and moves onto safe, unoccupied squares."""
    options = [Action.STAY] if _is_safe(state, state.npc.position, config) else []
    for action in MOVES:
        target = _neighbour(state, action)
        if target != state.agent.position and _is_safe(state, target, config):
            options.append(action)
    if not options:
        return Action.STAY  # boxed in: nowhere safe to go
    return options[rng.integers(len(options))]


def _fire_moves(state: GridworldState) -> list[Action]:
    """Moves onto adjacent fire, or none if the NPC is already on fire."""
    if state.tiles[state.npc.position] == Tile.FIRE:
        return []
    return [action for action in MOVES if state.tiles[_neighbour(state, action)] == Tile.FIRE]


def scripted_npc_policy(state: GridworldState, config: GridworldConfig, rng: np.random.Generator) -> Action:
    """The NPC's script: see the module docstring."""
    if not state.npc.alive:
        return Action.STAY

    if config.npc_fire_step_prob > 0:
        fire_moves = _fire_moves(state)
        if fire_moves and rng.random() < config.npc_fire_step_prob:
            return fire_moves[rng.integers(len(fire_moves))]

    if state.npc.energy < config.npc_hunger_threshold:
        food_move = _step_toward_food(state, config, rng)
        if food_move is not None:
            return food_move
    return _wander(state, config, rng)
