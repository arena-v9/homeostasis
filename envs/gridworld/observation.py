"""
What the agent sees: the whole map as a stack of (grid_size, grid_size) channels.
"""

import numpy as np

from envs.gridworld.config import GridworldConfig, Tile
from envs.gridworld.state import Body, GridworldState

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
    _add_body(obs, state.npc, config)
    return obs


def observe_npc_world(world: GridworldState, config: GridworldConfig) -> np.ndarray:
    """
    The NPC's own grid, in the parallel-grid setting (see episode.py): the same channels as `observe`, with the NPC
    on the shared `body` (and, if visible, `health` and `energy`) channels and nothing on `self`.
    """
    obs = _map_channels(world)
    _add_body(obs, world.npc, config)
    return obs


def observe_tv(world: GridworldState, config: GridworldConfig) -> np.ndarray:
    """
    One frame of the TV (see tv.py): the clip's map, with both its bodies (the pusher in the agent slot and the victim
    in the NPC slot) on the shared `body`, `health` and `energy` channels and nothing on `self`.
    """
    obs = _map_channels(world)
    for body in (world.agent, world.npc):
        _add_body(obs, body, config)
    return obs


def _map_channels(state: GridworldState) -> np.ndarray:
    """A fresh observation with the map filled in: walls, hazards and food."""
    obs = np.zeros((NUM_CHANNELS, state.grid_size, state.grid_size), dtype=np.float32)
    for channel, tile in TILE_CHANNELS.items():
        obs[CHANNEL_INDEX[channel]] = state.tiles == tile
    obs[CHANNEL_INDEX["food"]] = state.food
    return obs


def _add_body(obs: np.ndarray, body: Body, config: GridworldConfig) -> None:
    """Draws a living body other than the agent on the shared channels; its health and energy only if visible."""
    if not body.alive:
        return
    obs[CHANNEL_INDEX["body"]][body.position] = 1.0
    if config.npc_state_visible:
        obs[CHANNEL_INDEX["health"]][body.position] = body.health
        obs[CHANNEL_INDEX["energy"]][body.position] = body.energy
