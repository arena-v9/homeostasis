"""
One episode of the gridworld, in either setting:

  shared grid (default)   The agent and the NPC live on the same grid: the agent can push the NPC, they block each
                          other and compete for the same food.
  parallel grid           (`config.parallel_grid`) The NPC lives on its own grid, with its own fire and food. The two
                          grids never interact: they are separate simulations that share only the agent's view, which
                          shows them side by side as one (channels, grid_size, 2 * grid_size) image on the same
                          channels, the agent's grid on the left. Each grid keeps its wall border, so the seam is two
                          walls thick.

The batched environment, the replay recorder and the diagnostics all go through these functions, so the two settings
are handled in one place.
"""

from dataclasses import dataclass, replace

import numpy as np

from envs.gridworld.config import GridworldConfig
from envs.gridworld.dynamics import Action, StepResult, step_npc_world, take_step
from envs.gridworld.initial_state import generate_initial_state
from envs.gridworld.npc_policy import NpcPolicy, scripted_npc_policy
from envs.gridworld.observation import NUM_CHANNELS, observe, observe_npc_world
from envs.gridworld.state import Body, GridworldState


@dataclass
class Episode:
    state: GridworldState  # the agent's grid (with the NPC on it, in the shared-grid setting)
    npc_world: GridworldState | None = None  # the NPC's own grid, in the parallel-grid setting


def observation_shape(config: GridworldConfig) -> tuple[int, int, int]:
    width = 2 * config.grid_size if config.parallel_grid else config.grid_size
    return (NUM_CHANNELS, config.grid_size, width)


def new_episode(config: GridworldConfig, rng: np.random.Generator) -> Episode:
    if not config.parallel_grid:
        return Episode(generate_initial_state(config, rng))
    alone = replace(config, npc_enabled=False)
    return Episode(generate_initial_state(alone, rng), _new_npc_world(config, rng))


def _new_npc_world(config: GridworldConfig, rng: np.random.Generator) -> GridworldState:
    """
    A layout like the agent's (fire, food reachable without crossing fire), with the NPC where the agent would start.
    With npc_enabled False there's nobody on it. The agent slot holds a dead placeholder on a wall corner, which the
    NPC's script and movement treat as a wall.
    """
    layout = generate_initial_state(replace(config, npc_enabled=False), rng)
    npc = Body(layout.agent.position, 1.0, layout.agent.energy, alive=config.npc_enabled)
    return replace(layout, agent=Body((0, 0), alive=False), npc=npc)


def step_episode(
    episode: Episode,
    agent_action: Action,
    config: GridworldConfig,
    rng: np.random.Generator,
    npc_policy: NpcPolicy = scripted_npc_policy,
) -> tuple[Episode, StepResult]:
    """
    Advances the episode by one step. The result's reward, termination and agent flags come from the agent's grid; its
    NPC flags (on a hazard, died) from wherever the NPC lives. Doesn't modify `episode`.
    """
    if episode.npc_world is None:
        result = take_step(episode.state, agent_action, npc_policy(episode.state, config, rng), config, rng)
        return Episode(result.next_state), result

    npc_action = npc_policy(episode.npc_world, config, rng)
    result = take_step(episode.state, agent_action, Action.STAY, config, rng)  # no NPC on the agent's grid
    npc_world, npc_on_hazard, npc_died = step_npc_world(episode.npc_world, npc_action, config, rng)
    return Episode(result.next_state, npc_world), replace(result, npc_on_hazard=npc_on_hazard, npc_died=npc_died)


def observe_episode(episode: Episode, config: GridworldConfig) -> np.ndarray:
    """What the agent sees: its grid, and in the parallel-grid setting the NPC's grid to its right."""
    obs = observe(episode.state, config)
    if episode.npc_world is None:
        return obs
    return np.concatenate([obs, observe_npc_world(episode.npc_world, config)], axis=2)
