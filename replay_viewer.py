"""
Records gridworld episodes for the replay part of the viewer (ui/index.html), in the layout saved as a run's
replay.json (see run_outputs.py):

    episodes = record_episodes(act_fn, config, seeds=range(16), device="cpu")
    data = replay_data({"untrained": ..., "after 200k steps": episodes})

The layout:

    tiles                     tile name -> the number used in each episode's `tiles` grid
    checkpoints[]             one per training stage
      label
      episodes[]              one per layout seed
        seed, outcome ("survived", "died" or "discarded push"), return, fire_steps, food_eaten
        tiles                 the map, rows of tile numbers
        frames[]              one per step, starting with the initial state
          agent, npc          {"pos": [row, col], "health", "energy"}; npc is null when there is no living NPC
          food                [row, col] of each food item
          action, reward      what the agent did from this state and the reward it got (absent on the last frame)

`record_episodes` runs the single-env mechanics (gridworld.step) directly, so it records the true states, including
the final state of an episode that ends in death.
"""

from typing import Callable

import numpy as np
import torch as t

from batched_gridworld import NpcPolicy, npc_stays
from gridworld import Action, observe, step
from gridworld_config import GridworldConfig, Tile
from gridworld_layout import random_state
from gridworld_state import Body, GridworldState


def _body(body: Body) -> dict:
    return {"pos": [int(p) for p in body.pos], "health": round(body.health, 4), "energy": round(body.energy, 4)}


def _frame(state: GridworldState) -> dict:
    return {
        "agent": _body(state.agent),
        "npc": _body(state.npc) if state.npc.alive else None,
        "food": [[int(r), int(c)] for r, c in np.argwhere(state.food)],
    }


@t.inference_mode()
def record_episodes(
    act_fn: Callable[[t.Tensor], t.Tensor],
    config: GridworldConfig,
    seeds,
    device,
    npc_policy: NpcPolicy = npc_stays,
) -> list[dict]:
    """Runs one full episode per seed, choosing the agent's actions with `act_fn(obs_batch) -> actions` (all
    episodes advance together so the network sees one batch per step). Returns one dict per episode."""
    rngs = [np.random.default_rng(seed) for seed in seeds]
    states = [random_state(config, rng) for rng in rngs]
    frames = [[_frame(state)] for state in states]
    fire_steps, food_eaten = [0] * len(states), [0] * len(states)
    outcomes: list[str | None] = [None] * len(states)

    for _ in range(config.max_steps):
        active = [i for i, outcome in enumerate(outcomes) if outcome is None]
        if not active:
            break
        obs = t.from_numpy(np.stack([observe(states[i], config) for i in active])).to(device)
        actions = act_fn(obs).tolist()
        for i, action in zip(active, actions):
            res = step(states[i], Action(action), npc_policy(states[i], rngs[i]), config, rngs[i])
            frames[i][-1].update(action=Action(action).name.lower(), reward=round(res.reward, 4))
            if res.discarded:
                outcomes[i] = "discarded push"
                continue
            food_eaten[i] += int(res.state.agent.energy > states[i].agent.energy)  # energy only rises by eating
            fire_steps[i] += int(res.agent_on_hazard)
            frames[i].append(_frame(res.state))
            states[i] = res.state
            if res.terminated:
                outcomes[i] = "died"
            elif res.truncated:
                outcomes[i] = "survived"

    return [
        {
            "seed": int(seed),
            "outcome": outcomes[i] or "survived",
            "return": round(sum(f.get("reward", 0.0) for f in frames[i]), 2),
            "fire_steps": fire_steps[i],
            "food_eaten": food_eaten[i],
            "tiles": states[i].tiles.tolist(),
            "frames": frames[i],
        }
        for i, seed in enumerate(seeds)
    ]


def replay_data(checkpoints: dict[str, list[dict]]) -> dict:
    """`checkpoints` maps a label (e.g. "untrained", "after 100k steps") to the episodes recorded at that point."""
    return {
        "tiles": {tile.name.lower(): int(tile) for tile in Tile},
        "checkpoints": [{"label": label, "episodes": eps} for label, eps in checkpoints.items()],
    }
