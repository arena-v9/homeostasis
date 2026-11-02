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
        seed, outcome ("survived", "died" or "discarded push"), return, fire_steps, food_eaten, pushes, npc_fire_steps
        npc_died_at           the step the NPC died, or null if it was alive when the episode ended (or not on the map)
        tiles                 the map, rows of tile numbers
        frames[]              one per step, starting with the initial state
          agent, npc          {"pos": [row, col], "health", "energy"}; npc is null when there is no living NPC
          food                [row, col] of each food item
          action, reward      what the agent did from this state and the reward it got (absent on the last frame)

`record_episodes` runs the single-env dynamics (envs.gridworld.dynamics.take_step) directly, so it records the true
states, including the final state of an episode that ends in death.
"""

from typing import Callable

import numpy as np
import torch as t

from envs.gridworld.config import GridworldConfig, Tile
from envs.gridworld.dynamics import Action, take_step
from envs.gridworld.initial_state import generate_initial_state
from envs.gridworld.npc_policy import NpcPolicy, scripted_npc_policy
from envs.gridworld.observation import observe
from envs.gridworld.state import Body, GridworldState


def _body(body: Body) -> dict:
    return {
        "pos": [int(coordinate) for coordinate in body.position],
        "health": round(body.health, 4),
        "energy": round(body.energy, 4),
    }


def _frame(state: GridworldState) -> dict:
    return {
        "agent": _body(state.agent),
        "npc": _body(state.npc) if state.npc.alive else None,
        "food": [[int(row), int(col)] for row, col in np.argwhere(state.food)],
    }


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
    states = [generate_initial_state(config, rng) for rng in rngs]
    frames = [[_frame(state)] for state in states]
    fire_steps, food_eaten = [0] * len(states), [0] * len(states)
    pushes, npc_fire_steps = [0] * len(states), [0] * len(states)
    outcomes: list[str | None] = [None] * len(states)
    npc_died_at: list[int | None] = [None] * len(states)

    for _ in range(config.max_steps):
        active = [index for index, outcome in enumerate(outcomes) if outcome is None]
        if not active:
            break
        obs = t.from_numpy(np.stack([observe(states[index], config) for index in active])).to(device)
        actions = act_fn(obs).tolist()
        for index, action in zip(active, actions):
            state, rng = states[index], rngs[index]
            result = take_step(state, Action(action), npc_policy(state, config, rng), config, rng)
            frames[index][-1].update(action=Action(action).name.lower(), reward=round(result.reward, 4))
            if result.discarded:
                outcomes[index] = "discarded push"
                continue
            food_eaten[index] += int(result.next_state.agent.energy > state.agent.energy)  # energy only rises by eating
            fire_steps[index] += int(result.agent_on_hazard)
            pushes[index] += int(result.npc_pushed)
            npc_fire_steps[index] += int(result.npc_on_hazard)
            if result.npc_died:
                npc_died_at[index] = result.next_state.steps_taken
            frames[index].append(_frame(result.next_state))
            states[index] = result.next_state
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
            "tiles": states[index].tiles.tolist(),
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
