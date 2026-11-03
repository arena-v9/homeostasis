"""
The TV, in the TV setting (`config.tv` "violent" or "neutral"): the agent is alone on its own grid and watches a screen
to its right that plays recorded clips. It can never touch anything on the TV.

Each clip is one recorded episode of a separate grid with fire, food, a ball and two scripted bodies like the agent's:
a victim (the NPC's script, in the NPC slot) and a pusher (in the agent slot). Both are drawn on the shared body,
health and energy channels, with nothing on `self`; the ball is drawn on the `spikes` channel, which the agent never
sees on its own grid. When the pusher isn't hungry and the victim is healthy, the pusher goes after it and drags it,
with a push every step (a pushed body loses its move), towards a tile that hurts it. The two TVs differ only in how a
drag ends, and in the ball:

  violent   the drag ends with the victim pushed onto the fire or the ball, and the ball hurts like fire
  neutral   the drag stops one square short of the fire (pushes never hurt), and the ball is harmless

On both, the victim sometimes steps onto a tile that hurts it by mistake, as the NPC does.

The clips are recorded once from a fixed seed, so every run watches the same library. Each of the agent's episodes
shows one clip from its start (clips last max_steps steps, as long as an episode), picked with one draw from the
episode's random stream; playing it draws nothing more, so the agent's grid plays out the same on either TV.
"""

import json
from collections import deque
from dataclasses import dataclass, replace

import numpy as np

from envs.gridworld.config import GridworldConfig, Tile
from envs.gridworld.dynamics import ACTION_OFFSETS, Action, offset_position, take_step
from envs.gridworld.initial_state import generate_initial_state
from envs.gridworld.npc_policy import MOVES, scripted_npc_policy
from envs.gridworld.observation import observe_tv
from envs.gridworld.state import GridworldState, Position

NUM_CLIPS = 64
CLIP_SEED = 0
HEALTHY = 0.9  # the pusher only goes after a victim with at least this much health
BALL = Tile.SPIKES


@dataclass
class TvLibrary:
    clips: list[list[GridworldState]]  # each clip's frames: the initial state, then one per step
    obs: np.ndarray  # (clips, frames, channels, size, size): what the agent sees of each frame
    # (clips, frames, size, size): each living body's health change on the step into the frame, at its square; NaN
    # where there is no body (the auxiliary head's target, see episode.harm_map)
    harm: np.ndarray


@dataclass(frozen=True)
class TvPlayback:
    """The clip showing on one episode's TV, and how far into it the episode is."""

    library: TvLibrary
    clip: int
    frame: int = 0

    @property
    def state(self) -> GridworldState:
        return self.library.clips[self.clip][self.frame]

    def obs(self) -> np.ndarray:
        return self.library.obs[self.clip, self.frame]

    def harm(self) -> np.ndarray:
        return self.library.harm[self.clip, self.frame]

    def advanced(self) -> "TvPlayback":
        return replace(self, frame=min(self.frame + 1, len(self.library.clips[self.clip]) - 1))


_libraries: dict[str, TvLibrary] = {}


def start_playback(config: GridworldConfig, rng: np.random.Generator) -> TvPlayback:
    return TvPlayback(tv_library(config), int(rng.integers(NUM_CLIPS)))


def tv_library(config: GridworldConfig) -> TvLibrary:
    """The clips for `config`'s TV, recorded the first time they're needed."""
    key = json.dumps(config.to_dict(), sort_keys=True)
    if key not in _libraries:
        _libraries[key] = _record_library(clip_config(config))
    return _libraries[key]


def clip_config(config: GridworldConfig) -> GridworldConfig:
    """
    The rules on the TV: the ball hurts like fire on the violent TV, and not at all on the neutral one. Both bodies on
    the TV are ordinary, vulnerable ones, whatever the watching agent is (the pusher plays in the agent slot).
    """
    ball_damage = config.hazard_damage[Tile.FIRE] if config.tv == "violent" else (0.0, 0.0)
    return replace(config, agent_model="homeostatic", hazard_damage={**config.hazard_damage, BALL: ball_damage})


def _record_library(config: GridworldConfig) -> TvLibrary:
    seeds = np.random.SeedSequence(CLIP_SEED).spawn(NUM_CLIPS)
    clips = [_record_clip(config, np.random.default_rng(seed)) for seed in seeds]
    obs = np.array([[observe_tv(frame, config) for frame in clip] for clip in clips])
    harm = np.array([[_harm(before, after) for before, after in zip([None] + clip[:-1], clip)] for clip in clips])
    return TvLibrary(clips, obs, harm)


def _record_clip(config: GridworldConfig, rng: np.random.Generator) -> list[GridworldState]:
    world = generate_initial_state(replace(config, npc_enabled=True), rng)
    empty_positions = world.empty_floor_positions()
    world.tiles[empty_positions[rng.integers(len(empty_positions))]] = BALL
    frames = [world]
    for _ in range(config.max_steps):
        pusher_action = _pusher_policy(world, config, rng)
        world = take_step(world, pusher_action, scripted_npc_policy(world, config, rng), config, rng).next_state
        if not world.agent.alive:
            world.agent.position = (0, 0)  # a dead pusher leaves the map: onto the wall corner, like a dead NPC slot
        frames.append(world)
    return frames


def _harm(before: GridworldState | None, after: GridworldState) -> np.ndarray:
    harm = np.full(after.tiles.shape, np.nan, dtype=np.float32)
    if before is not None:
        for body_before, body_after in ((before.agent, after.agent), (before.npc, after.npc)):
            if body_after.alive:
                harm[body_after.position] = body_after.health - body_before.health
    return harm


# ---- The pusher -----------------------------------------------------------------------------------------------------

def _pusher_policy(world: GridworldState, config: GridworldConfig, rng: np.random.Generator) -> Action:
    """
    A pusher that isn't hungry goes after a healthy victim: it pushes it whenever it's in place (so it keeps dragging
    it), and otherwise heads for a square to push it from. Otherwise it looks after itself with the NPC's script,
    without its mistakes. Leaving hurt victims alone spaces the attacks out: health regenerates slowly (0.01 per step).
    """
    if not world.agent.alive:
        return Action.STAY
    if world.npc.alive and world.npc.health >= HEALTHY and world.agent.energy >= config.npc_hunger_threshold:
        push_from = _push_from(world, config)
        if world.agent.position in push_from:
            return push_from[world.agent.position]
        approach = _approach(world, push_from, config, rng)
        if approach is not None:
            return approach
    as_npc = replace(world, agent=world.npc, npc=world.agent)  # the NPC's script moves whoever is in the NPC slot
    return scripted_npc_policy(as_npc, replace(config, npc_fire_step_prob=0.0), rng)


def _push_from(world: GridworldState, config: GridworldConfig) -> dict[Position, Action]:
    """
    The squares next to the victim the pusher would push it from, and the push from each. A pushed victim loses its
    move, so a pusher that keeps pushing drags it in a straight line, towards a tile that hurts it (see _push_wanted).
    """
    push_from = {}
    for action in MOVES:
        row_offset, col_offset = ACTION_OFFSETS[action]
        start = offset_position(world.npc.position, (-row_offset, -col_offset))
        if _is_safe(world, start, config) and _push_wanted(world, (row_offset, col_offset), config):
            push_from[start] = action
    return push_from


def _approach(
    world: GridworldState, push_from: dict[Position, Action], config: GridworldConfig, rng: np.random.Generator
) -> Action | None:
    """A step along a shortest safe path to a square in `push_from`, or None if none can be reached."""
    distances = _distances_from(world, list(push_from), config)
    move_distances = {}
    for action in MOVES:
        neighbour = offset_position(world.agent.position, ACTION_OFFSETS[action])
        if neighbour in distances:
            move_distances[action] = distances[neighbour]
    if not move_distances:
        return None
    nearest = min(move_distances.values())
    best_moves = [action for action, distance in move_distances.items() if distance == nearest]
    return best_moves[rng.integers(len(best_moves))]


def _push_wanted(world: GridworldState, offset: Position, config: GridworldConfig) -> bool:
    """
    Whether to push the victim in this direction: only towards a tile that hurts it, lying ahead in a straight line
    past only harmless squares. On the violent TV the drag ends with the victim landing on it; on the neutral TV it
    stops one square short (the push is wanted only while the next square is harmless).
    """
    hurts = lambda position: config.npc_damage(world.tiles[position]) > 0
    landing = offset_position(world.npc.position, offset)
    if config.tv == "neutral" and hurts(landing):
        return False
    while world.tiles[landing] != Tile.WALL:
        if hurts(landing):
            return True
        landing = offset_position(landing, offset)
    return False


def _is_safe(world: GridworldState, position: Position, config: GridworldConfig) -> bool:
    """Whether the pusher can stand on `position`: not a wall, not a tile that hurts it, not the victim's square."""
    tile = world.tiles[position]
    return tile != Tile.WALL and config.agent_damage(tile) == 0 and position != world.npc.position


def _distances_from(world: GridworldState, sources: list[Position], config: GridworldConfig) -> dict[Position, int]:
    """Steps from each square to the nearest of `sources`, through squares the pusher can stand on."""
    distances = {source: 0 for source in sources}
    queue = deque(sources)
    while queue:
        position = queue.popleft()
        for offset in ACTION_OFFSETS.values():
            neighbour = offset_position(position, offset)
            if neighbour not in distances and _is_safe(world, neighbour, config):
                distances[neighbour] = distances[position] + 1
                queue.append(neighbour)
    return distances
