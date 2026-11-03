"""
One episode of the gridworld, in any of its settings:

  shared grid (default)   The agent and the NPC live on the same grid: the agent can push the NPC, they block each
                          other and compete for the same food.
  parallel grid           (`config.parallel_grid`) The NPC lives on its own grid, with its own fire and food. The two
                          grids never interact: they are separate simulations that share only the agent's view, which
                          shows them side by side as one (channels, grid_size, 2 * grid_size) image on the same
                          channels, the agent's grid on the left. Each grid keeps its wall border, so the seam is two
                          walls thick. With `config.npc_on_agent_grid` the NPC lives on the agent's grid instead
                          (as in the shared grid) and the second grid stays empty: the push test.
  TV                      (`config.tv`) The agent is alone on its grid, and to its right, where the parallel grid would
                          be, a TV plays a recorded clip of a pusher and a victim (see tv.py).

The batched environment, the replay recorder and the diagnostics all go through these functions, so the settings are
handled in one place.
"""

from dataclasses import dataclass, replace

import numpy as np

from envs.gridworld.config import GridworldConfig
from envs.gridworld.dynamics import Action, StepResult, step_npc_world, take_step
from envs.gridworld.initial_state import generate_initial_state
from envs.gridworld.npc_policy import NpcPolicy, scripted_npc_policy
from envs.gridworld.observation import NUM_CHANNELS, observe, observe_npc_world
from envs.gridworld.state import Body, GridworldState
from envs.gridworld.tv import TvPlayback, start_playback


@dataclass
class Episode:
    state: GridworldState  # the agent's grid (with the NPC on it, in the shared-grid setting)
    npc_world: GridworldState | None = None  # the NPC's own grid, in the parallel-grid setting
    tv: TvPlayback | None = None  # the clip on the TV, in the TV setting


def observation_shape(config: GridworldConfig) -> tuple[int, int, int]:
    two_grids = config.parallel_grid or config.tv != "off"
    width = 2 * config.grid_size if two_grids else config.grid_size
    return (NUM_CHANNELS, config.grid_size, width)


def new_episode(config: GridworldConfig, rng: np.random.Generator) -> Episode:
    alone = replace(config, npc_enabled=False)
    if config.tv != "off":
        return Episode(generate_initial_state(alone, rng), tv=start_playback(config, rng))
    if not config.parallel_grid:
        return Episode(generate_initial_state(config, rng))
    if config.npc_on_agent_grid:
        return Episode(generate_initial_state(config, rng), _new_npc_world(alone, rng))
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
    NPC flags (on a hazard, died) from wherever the NPC lives (never set in the TV setting). Doesn't modify `episode`.
    """
    if episode.tv is not None:
        result = take_step(episode.state, agent_action, Action.STAY, config, rng)  # nobody else on the agent's grid
        return Episode(result.next_state, tv=episode.tv.advanced()), result

    if episode.npc_world is None or config.npc_on_agent_grid:  # the NPC on the agent's grid; any second grid is empty
        result = take_step(episode.state, agent_action, npc_policy(episode.state, config, rng), config, rng)
        return Episode(result.next_state, episode.npc_world), result

    npc_action = npc_policy(episode.npc_world, config, rng)
    result = take_step(episode.state, agent_action, Action.STAY, config, rng)  # no NPC on the agent's grid
    npc_world, npc_on_hazard, npc_died = step_npc_world(episode.npc_world, npc_action, config, rng)
    return Episode(result.next_state, npc_world), replace(result, npc_on_hazard=npc_on_hazard, npc_died=npc_died)


def second_grid(episode: Episode) -> GridworldState | None:
    """What the agent sees to the right of its grid: the NPC's grid, the TV's current frame, or nothing."""
    if episode.tv is not None:
        return episode.tv.state
    return episode.npc_world


def observe_episode(episode: Episode, config: GridworldConfig) -> np.ndarray:
    """What the agent sees: its grid, and in the parallel-grid and TV settings the second grid to its right."""
    obs = observe(episode.state, config)
    if episode.tv is not None:
        return np.concatenate([obs, episode.tv.obs()], axis=2)
    if episode.npc_world is not None:
        return np.concatenate([obs, observe_npc_world(episode.npc_world, config)], axis=2)
    return obs


def harm_map(episode: Episode, next_episode: Episode, config: GridworldConfig) -> np.ndarray:
    """
    How much each body's health changed on the step from `episode` to `next_episode`, at its square in the next
    observation, and NaN where there's no body: the auxiliary head's target (see DQNTrainer.harm_loss). Every body
    counts, the agent's own included (also on the step that kills it), the NPC wherever it lives and both bodies on
    the TV.
    """
    size = config.grid_size
    harm = np.full(observation_shape(config)[1:], np.nan, dtype=np.float32)
    before, after = episode.state, next_episode.state
    harm[after.agent.position] = after.agent.health - before.agent.health
    if before.npc.alive and after.npc.alive:
        harm[after.npc.position] = after.npc.health - before.npc.health
    if next_episode.npc_world is not None and next_episode.npc_world.npc.alive:
        row, col = next_episode.npc_world.npc.position
        harm[row, size + col] = next_episode.npc_world.npc.health - episode.npc_world.npc.health
    if next_episode.tv is not None:
        harm[:, size:] = next_episode.tv.harm()
    return harm
