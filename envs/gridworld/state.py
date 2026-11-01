"""
The complete state of one gridworld episode. It's a plain dataclass so tests and the evaluation harness can build
exact scenarios by hand (e.g. "agent left of the NPC, fire directly behind it") and pass them to `step` / `observe`.
"""

from dataclasses import dataclass, field, replace

import numpy as np

from envs.gridworld.config import Tile

Pos = tuple[int, int]  # (row, col), row 0 at the top


@dataclass
class Body:
    pos: Pos
    health: float = 1.0
    energy: float = 1.0
    alive: bool = True


@dataclass
class GridworldState:
    tiles: np.ndarray  # (size, size) array of Tile values
    food: np.ndarray  # (size, size) bool: True where a food item currently sits
    agent: Body
    npc: Body
    food_timers: list[int] = field(default_factory=list)  # steps left until each eaten item respawns
    t: int = 0  # steps taken so far this episode

    @property
    def size(self) -> int:
        return self.tiles.shape[0]

    def copy(self) -> "GridworldState":
        return GridworldState(
            tiles=self.tiles.copy(),
            food=self.food.copy(),
            agent=replace(self.agent),
            npc=replace(self.npc),
            food_timers=list(self.food_timers),
            t=self.t,
        )

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
        size = len(grid)
        assert all(len(row) == size for row in grid), "Map must be square"
        tiles = np.full((size, size), Tile.FLOOR, dtype=np.int8)
        food = np.zeros((size, size), dtype=bool)
        agent_pos = npc_pos = None
        symbols = {"#": Tile.WALL, "X": Tile.FIRE, "S": Tile.SPIKES, "C": Tile.COLD}
        for r, row in enumerate(grid):
            for c, ch in enumerate(row):
                if ch in symbols:
                    tiles[r, c] = symbols[ch]
                elif ch == "F":
                    food[r, c] = True
                elif ch == "A":
                    agent_pos = (r, c)
                elif ch == "N":
                    npc_pos = (r, c)
                else:
                    assert ch == ".", f"Unknown map symbol {ch!r}"
        assert agent_pos is not None and npc_pos is not None, "Map needs one A and one N"
        return cls(
            tiles=tiles,
            food=food,
            agent=Body(agent_pos, agent_health, agent_energy),
            npc=Body(npc_pos, npc_health, npc_energy),
        )
