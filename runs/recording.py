"""
Records gridworld episodes for the replay part of the viewer (ui/index.html), in the layout saved as a run's
replay.json (see runs/outputs.py):

    episodes = record_episodes(act_fn, config, seeds=range(16), device="cpu")
    data = replay_data({"untrained": ..., "after 200k steps": episodes})

The layout:

    tiles                     tile name -> the number used in each episode's `tiles` grid
    checkpoints[]             one per training stage
      label
      episodes[]              one per layout seed
        seed, outcome ("survived" or "died"), return, fire_steps, food_eaten, pushes, npc_fire_steps
        npc_died_at           the step the NPC died, or null if it was alive when the episode ended (or not on the map)
        tiles                 the map, rows of tile numbers
        npc_tiles             the second map to the right of the agent's: the NPC's own map in the parallel-grid
                              setting, the TV clip's map in the TV setting (absent otherwise)
        tv                    true in the TV setting
        frames[]              one per step, starting with the initial state
          agent, npc          {"pos": [row, col], "health", "energy"}; npc is null when there is no living NPC. In the
                              parallel-grid setting the NPC's position is on its own map; in the TV setting npc is
                              the TV's victim
          pusher              the TV's pusher, in the TV setting (null once it has died)
          food                [row, col] of each food item
          npc_food            food on the second map
          action, reward      what the agent did from this state and the reward it got (absent on the last frame)

`record_episodes` runs single episodes (envs.gridworld.episode) directly, so it records the true states, including
the final state of an episode that ends in death.
"""

from typing import Callable

import numpy as np
import torch as t

from envs.gridworld.config import GridworldConfig, Tile
from envs.gridworld.dynamics import Action
from envs.gridworld.episode import Episode, new_episode, observe_episode, second_grid, step_episode
from envs.gridworld.npc_policy import NpcPolicy, scripted_npc_policy
from envs.gridworld.state import Body, GridworldState


def _body(body: Body) -> dict:
    return {
        "pos": [int(coordinate) for coordinate in body.position],
        "health": round(body.health, 4),
        "energy": round(body.energy, 4),
    }


def _food(state: GridworldState) -> list[list[int]]:
    return [[int(row), int(col)] for row, col in np.argwhere(state.food)]


def _frame(episode: Episode) -> dict:
    second = second_grid(episode)
    npc_home = second or episode.state  # where the NPC (or the TV's victim) lives
    frame = {
        "agent": _body(episode.state.agent),
        "npc": _body(npc_home.npc) if npc_home.npc.alive else None,
        "food": _food(episode.state),
    }
    if second is not None:
        frame["npc_food"] = _food(second)
    if episode.tv is not None:
        frame["pusher"] = _body(second.agent) if second.agent.alive else None
    return frame


@t.inference_mode()
def record_episodes(
    act_fn: Callable[[t.Tensor], t.Tensor],
    config: GridworldConfig,
    seeds,
    device,
    npc_policy: NpcPolicy = scripted_npc_policy,
) -> list[dict]:
    """Runs one full episode per seed, choosing the agent's actions with `act_fn(obs_batch) -> actions` (all
    episodes advance together so the network sees one batch per step). Returns one dict per episode."""
    rngs = [np.random.default_rng(seed) for seed in seeds]
    episodes = [new_episode(config, rng) for rng in rngs]
    frames = [[_frame(episode)] for episode in episodes]
    fire_steps, food_eaten = [0] * len(episodes), [0] * len(episodes)
    pushes, npc_fire_steps = [0] * len(episodes), [0] * len(episodes)
    outcomes: list[str | None] = [None] * len(episodes)
    npc_died_at: list[int | None] = [None] * len(episodes)

    for _ in range(config.max_steps):
        active = [index for index, outcome in enumerate(outcomes) if outcome is None]
        if not active:
            break
        obs = t.from_numpy(np.stack([observe_episode(episodes[index], config) for index in active])).to(device)
        actions = act_fn(obs).tolist()
        for index, action in zip(active, actions):
            episode, rng = episodes[index], rngs[index]
            next_episode, result = step_episode(episode, Action(action), config, rng, npc_policy)
            frames[index][-1].update(action=Action(action).name.lower(), reward=round(result.reward, 4))
            food_eaten[index] += int(result.agent_ate)
            fire_steps[index] += int(result.agent_on_hazard)
            pushes[index] += int(result.npc_pushed)
            npc_fire_steps[index] += int(result.npc_on_hazard)
            frames[index].append(_frame(next_episode))
            if frames[index][-2]["npc"] and not frames[index][-1]["npc"]:
                npc_died_at[index] = len(frames[index]) - 1
            episodes[index] = next_episode
            if result.terminated:
                outcomes[index] = "died"
            elif result.truncated:
                outcomes[index] = "survived"

    return [
        {
            "seed": int(seed),
            "outcome": outcomes[index] or "survived",
            "return": round(sum(frame.get("reward", 0.0) for frame in frames[index]), 2),
            "fire_steps": fire_steps[index],
            "food_eaten": food_eaten[index],
            "pushes": pushes[index],
            "npc_fire_steps": npc_fire_steps[index],
            "npc_died_at": npc_died_at[index],
            "tiles": episodes[index].state.tiles.tolist(),
            **({"npc_tiles": second_grid(episodes[index]).tiles.tolist()} if second_grid(episodes[index]) else {}),
            **({"tv": True} if episodes[index].tv is not None else {}),
            "frames": frames[index],
        }
        for index, seed in enumerate(seeds)
    ]


def replay_data(checkpoints: dict[str, list[dict]]) -> dict:
    """`checkpoints` maps a label (e.g. "untrained", "after 100k steps") to the episodes recorded at that point."""
    return {
        "tiles": {tile.name.lower(): int(tile) for tile in Tile},
        "checkpoints": [{"label": label, "episodes": episodes} for label, episodes in checkpoints.items()],
    }
