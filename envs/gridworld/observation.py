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
    obs = _map_channels(state)
    obs[CHANNEL_INDEX["body"]][state.agent.position] = 1.0
    obs[CHANNEL_INDEX["self"]][state.agent.position] = 1.0
    obs[CHANNEL_INDEX["health"]][state.agent.position] = state.agent.health
    obs[CHANNEL_INDEX["energy"]][state.agent.position] = state.agent.energy
    _add_npc(obs, state, config)
    return obs


def observe_npc_world(world: GridworldState, config: GridworldConfig) -> np.ndarray:
    """
    The NPC's own grid, in the parallel-grid setting (see episode.py): the same channels as `observe`, with the NPC
    on the shared `body` (and, if visible, `health` and `energy`) channels and nothing on `self`.
    """
    obs = _map_channels(world)
    _add_npc(obs, world, config)
    return obs


def _map_channels(state: GridworldState) -> np.ndarray:
    """A fresh observation with the map filled in: walls, hazards and food."""
    obs = np.zeros((NUM_CHANNELS, state.grid_size, state.grid_size), dtype=np.float32)
    for channel, tile in TILE_CHANNELS.items():
        obs[CHANNEL_INDEX[channel]] = state.tiles == tile
    obs[CHANNEL_INDEX["food"]] = state.food
    return obs


def _add_npc(obs: np.ndarray, state: GridworldState, config: GridworldConfig) -> None:
    """Draws a living NPC on the shared channels; its health and energy only if they're visible."""
    if not state.npc.alive:
        return
    obs[CHANNEL_INDEX["body"]][state.npc.position] = 1.0
    if config.npc_state_visible:
        obs[CHANNEL_INDEX["health"]][state.npc.position] = state.npc.health
        obs[CHANNEL_INDEX["energy"]][state.npc.position] = state.npc.energy
