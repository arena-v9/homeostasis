"""
Random starting states for the gridworld. This is the minimal version: fire, food and bodies go on random floor
tiles, with a check that all food is reachable without crossing fire.
"""

from collections import deque

import numpy as np

from envs.gridworld.config import GridworldConfig, Tile
from envs.gridworld.mechanics import ACTION_DELTAS, empty_floor_tiles
from envs.gridworld.state import Body, GridworldState, Pos


def _reachable(tiles: np.ndarray, start: Pos) -> set[Pos]:
    """Squares reachable from `start` without walking through walls or fire."""
    seen, queue = {start}, deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in ACTION_DELTAS.values():
            nxt = (r + dr, c + dc)
            if nxt not in seen and tiles[nxt] not in (Tile.WALL, Tile.FIRE):
                seen.add(nxt)
                queue.append(nxt)
    return seen


def _pick(rng: np.random.Generator, options: list[Pos]) -> Pos:
    return options[rng.integers(len(options))]


def random_state(config: GridworldConfig, rng: np.random.Generator, max_tries: int = 100) -> GridworldState:
    """A fresh episode: walls around the border, `num_fire` fire tiles, `num_food` food items, the agent (and the NPC,
    if enabled) on empty floor. Bodies start at full health with energy drawn from `start_energy_range`."""
    size = config.size
    for _ in range(max_tries):
        tiles = np.full((size, size), Tile.FLOOR, dtype=np.int8)
        tiles[0, :] = tiles[-1, :] = tiles[:, 0] = tiles[:, -1] = Tile.WALL
        floor = [(r, c) for r in range(1, size - 1) for c in range(1, size - 1)]
        for i in rng.choice(len(floor), size=config.num_fire, replace=False):
            tiles[floor[i]] = Tile.FIRE

        # Bodies go down first (a dead NPC is a placeholder on the wall corner, never shown or used)
        state = GridworldState(
            tiles=tiles,
            food=np.zeros((size, size), dtype=bool),
            agent=Body((0, 0)),
            npc=Body((0, 0), alive=False),
        )
        state.agent = Body(_pick(rng, empty_floor_tiles(state)), 1.0, float(rng.uniform(*config.start_energy_range)))
        if config.npc_enabled:
            state.npc = Body(_pick(rng, empty_floor_tiles(state)), 1.0, float(rng.uniform(*config.start_energy_range)))
        for _ in range(config.num_food):
            state.food[_pick(rng, empty_floor_tiles(state))] = True

        reachable = _reachable(tiles, state.agent.pos)
        food_positions = {(int(r), int(c)) for r, c in np.argwhere(state.food)}
        if food_positions <= reachable and (not config.npc_enabled or state.npc.pos in reachable):
            return state
    raise RuntimeError(f"Couldn't generate a layout with reachable food in {max_tries} tries")
