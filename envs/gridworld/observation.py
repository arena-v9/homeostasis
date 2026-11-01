"""
What the agent sees: the whole map as a stack of (grid_size, grid_size) channels.
"""

import numpy as np

from envs.gridworld.config import GridworldConfig, Tile
from envs.gridworld.state import GridworldState

# Observation channels, in order
CHANNELS = ("wall", "food", "fire", "spikes", "cold", "body", "self", "health", "energy")
NUM_CHANNELS = len(CHANNELS)
CHANNEL_INDEX = {name: index for index, name in enumerate(CHANNELS)}
TILE_CHANNELS = {"wall": Tile.WALL, "fire": Tile.FIRE, "spikes": Tile.SPIKES, "cold": Tile.COLD}


def observe(state: GridworldState, config: GridworldConfig) -> np.ndarray:
    """
    The agent's view of the whole map: a (NUM_CHANNELS, grid_size, grid_size) float32 array with channels CHANNELS.

    Both bodies appear on the same `body`, `health` and `energy` channels - that shared representation is what the
    experiment tests. Only `self` tells them apart. In the hidden condition (`npc_state_visible=False`), `health`
    and `energy` are 0 at the NPC's square. A dead NPC is not shown at all.
    """
    obs = np.zeros((NUM_CHANNELS, state.grid_size, state.grid_size), dtype=np.float32)

    for channel, tile in TILE_CHANNELS.items():
        obs[CHANNEL_INDEX[channel]] = state.tiles == tile
    obs[CHANNEL_INDEX["food"]] = state.food

    living_bodies = [state.agent] + ([state.npc] if state.npc.alive else [])
    for body in living_bodies:
        obs[CHANNEL_INDEX["body"]][body.position] = 1.0
    obs[CHANNEL_INDEX["self"]][state.agent.position] = 1.0

    obs[CHANNEL_INDEX["health"]][state.agent.position] = state.agent.health
    obs[CHANNEL_INDEX["energy"]][state.agent.position] = state.agent.energy
    if state.npc.alive and config.npc_state_visible:
        obs[CHANNEL_INDEX["health"]][state.npc.position] = state.npc.health
        obs[CHANNEL_INDEX["energy"]][state.npc.position] = state.npc.energy

    return obs
