"""
Random starting states for the gridworld. This is the minimal version: fire, food and bodies go on random floor
squares, with a check that all food (and the NPC) is reachable without crossing fire.
"""

from collections import deque

import numpy as np

from envs.gridworld.config import GridworldConfig, Tile
from envs.gridworld.dynamics import ACTION_OFFSETS, offset_position
from envs.gridworld.state import Body, GridworldState, Position


def _positions_reachable_from(tiles: np.ndarray, start: Position) -> set[Position]:
    """Squares reachable from `start` without walking through walls or fire (breadth-first search)."""
    reached, queue = {start}, deque([start])
    while queue:
        position = queue.popleft()
        for offset in ACTION_OFFSETS.values():
            neighbour = offset_position(position, offset)
            if neighbour not in reached and tiles[neighbour] not in (Tile.WALL, Tile.FIRE):
                reached.add(neighbour)
                queue.append(neighbour)
    return reached


def _random_empty_position(state: GridworldState, rng: np.random.Generator) -> Position:
    empty_positions = state.empty_floor_positions()
    return empty_positions[rng.integers(len(empty_positions))]


def generate_initial_state(config: GridworldConfig, rng: np.random.Generator, max_tries: int = 100) -> GridworldState:
    """
    A fresh episode: walls around the border, `num_fire` fire tiles, `num_food` food items, and the agent (and the
    NPC, if enabled) on empty floor. Bodies start at full health with energy drawn from `start_energy_range`.
    Layouts where some food or the NPC can't be reached without crossing fire are redrawn.
    """
    grid_size = config.grid_size
    for _ in range(max_tries):
        tiles = np.full((grid_size, grid_size), Tile.FLOOR, dtype=np.int8)
        tiles[0, :] = tiles[-1, :] = tiles[:, 0] = tiles[:, -1] = Tile.WALL
        floor_positions = [(row, col) for row in range(1, grid_size - 1) for col in range(1, grid_size - 1)]
        for index in rng.choice(len(floor_positions), size=config.num_fire, replace=False):
            tiles[floor_positions[index]] = Tile.FIRE

        # Bodies go down first (a dead NPC is a placeholder on the wall corner, never shown or used)
        state = GridworldState(
            tiles=tiles,
            food=np.zeros((grid_size, grid_size), dtype=bool),
            agent=Body((0, 0)),
            npc=Body((0, 0), alive=False),
        )
        state.agent = Body(_random_empty_position(state, rng), 1.0, float(rng.uniform(*config.start_energy_range)))
        if config.npc_enabled:
            state.npc = Body(_random_empty_position(state, rng), 1.0, float(rng.uniform(*config.start_energy_range)))
        for _ in range(config.num_food):
            state.food[_random_empty_position(state, rng)] = True

        reachable = _positions_reachable_from(tiles, state.agent.position)
        food_positions = {(int(row), int(col)) for row, col in np.argwhere(state.food)}
        if food_positions <= reachable and (not config.npc_enabled or state.npc.position in reachable):
            return state
    raise RuntimeError(f"Couldn't generate a layout with reachable food in {max_tries} tries")
