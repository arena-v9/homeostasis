"""
The complete state of one gridworld episode. It's a plain dataclass so tests and the evaluation harness can build
exact scenarios by hand (e.g. "agent left of the NPC, fire directly behind it") and pass them to `take_step` /
`observe`.
"""

from dataclasses import dataclass, field, replace

import numpy as np

from envs.gridworld.config import Tile

Position = tuple[int, int]  # (row, col), row 0 at the top


@dataclass
class Body:
    """One body (the agent or the NPC): where it is and its two internal variables, both in [0, 1]."""

    position: Position
    health: float = 1.0
    energy: float = 1.0
    alive: bool = True


@dataclass
class GridworldState:
    tiles: np.ndarray  # (grid_size, grid_size) array of Tile values
    food: np.ndarray  # (grid_size, grid_size) bool: True where a food item currently sits
    agent: Body
    npc: Body
    food_respawn_timers: list[int] = field(default_factory=list)  # steps left until each eaten item respawns
    steps_taken: int = 0  # steps taken so far this episode

    @property
    def grid_size(self) -> int:
        return self.tiles.shape[0]

    def copy(self) -> "GridworldState":
        return GridworldState(
            tiles=self.tiles.copy(),
            food=self.food.copy(),
            agent=replace(self.agent),
            npc=replace(self.npc),
            food_respawn_timers=list(self.food_respawn_timers),
            steps_taken=self.steps_taken,
        )

    def empty_floor_positions(self) -> list[Position]:
        """
        Floor squares with no food and no living body on them: where new food or bodies can be placed.
        """
        occupied = {self.agent.position} | ({self.npc.position} if self.npc.alive else set())
        rows, cols = np.nonzero((self.tiles == Tile.FLOOR) & ~self.food)
        positions = [(int(row), int(col)) for row, col in zip(rows, cols)]
        return [position for position in positions if position not in occupied]

    @classmethod
    def from_ascii(
        cls,
        rows: list[str],
        agent_health: float = 1.0,
        agent_energy: float = 1.0,
        npc_health: float = 1.0,
        npc_energy: float = 1.0,
    ) -> "GridworldState":
        """
        Builds a state from a picture of the map:

            #  wall     .  floor     F  food     X  fire     S  spikes     C  cold     A  agent     N  NPC

        e.g. (moving right would push the NPC onto fire):
            ["# # # # # # #",
             "# . . . . F #",
             "# . . . . . #",
             "# A N X . . #",
             "# . . . . . #",
             "# F . . . . #",
             "# # # # # # #"]
        Spaces are ignored. Agents, NPCs and food stand on floor.
        """
        grid = [row.replace(" ", "") for row in rows]
        grid_size = len(grid)
        assert all(len(row) == grid_size for row in grid), "Map must be square"
        tiles = np.full((grid_size, grid_size), Tile.FLOOR, dtype=np.int8)
        food = np.zeros((grid_size, grid_size), dtype=bool)
        agent_position = npc_position = None
        tile_symbols = {"#": Tile.WALL, "X": Tile.FIRE, "S": Tile.SPIKES, "C": Tile.COLD}
        for row_index, row in enumerate(grid):
            for col_index, symbol in enumerate(row):
                position = (row_index, col_index)
                if symbol in tile_symbols:
                    tiles[position] = tile_symbols[symbol]
                elif symbol == "F":
                    food[position] = True
                elif symbol == "A":
                    agent_position = position
                elif symbol == "N":
                    npc_position = position
                else:
                    assert symbol == ".", f"Unknown map symbol {symbol!r}"
        assert agent_position is not None and npc_position is not None, "Map needs one A and one N"
        return cls(
            tiles=tiles,
            food=food,
            agent=Body(agent_position, agent_health, agent_energy),
            npc=Body(npc_position, npc_health, npc_energy),
        )
