"""
Records gridworld episodes and writes them to a self-contained HTML replay page (built from the template in
replay_viewer.html): open it in any browser to play all recorded episodes side by side, each with the agent's
health and energy bars above its grid, and switch between training stages.

    episodes = record_episodes(act_fn, config, seeds=range(16), device="cpu")
    write_replay_html({"after 200k steps": episodes}, "outputs/replay.html")

`record_episodes` runs the single-env mechanics (gridworld.step) directly, so it records the true states, including
the final state of an episode that ends in death.
"""

import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch as t

from batched_gridworld import NpcPolicy, npc_stays
from gridworld import Action, observe, step
from gridworld_config import GridworldConfig, Tile
from gridworld_layout import random_state
from gridworld_state import GridworldState

# The page itself; `write_replay_html` fills in __TITLE__ and __DATA__ (the recorded episodes as JSON)
TEMPLATE_PATH = Path(__file__).with_name("replay_viewer.html")


def _frame(state: GridworldState) -> dict:
    npc = state.npc
    return {
        "agent": list(state.agent.pos),
        "h": round(state.agent.health, 4),
        "e": round(state.agent.energy, 4),
        "npc": [*npc.pos, round(npc.health, 4), round(npc.energy, 4)] if npc.alive else None,
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
    episodes = [
        {"seed": int(seed), "tiles": state.tiles.tolist(), "frames": [_frame(state)], "fire_steps": 0, "food_eaten": 0}
        for seed, state in zip(seeds, states)
    ]
    done = [False] * len(states)

    for _ in range(config.max_steps):
        active = [i for i, d in enumerate(done) if not d]
        if not active:
            break
        obs = t.from_numpy(np.stack([observe(states[i], config) for i in active])).to(device)
        actions = act_fn(obs).tolist()
        for i, action in zip(active, actions):
            res = step(states[i], Action(action), npc_policy(states[i], rngs[i]), config, rngs[i])
            ep = episodes[i]
            ep["frames"][-1].update(a=Action(action).name.lower(), r=round(res.reward, 4))
            if res.discarded:
                ep["outcome"] = "discarded push"
                done[i] = True
                continue
            ep["food_eaten"] += int(res.state.agent.energy > states[i].agent.energy)  # energy only rises by eating
            ep["fire_steps"] += int(res.agent_on_hazard)
            ep["frames"].append(_frame(res.state))
            states[i] = res.state
            if res.terminated:
                ep["outcome"] = "died"
                done[i] = True
            elif res.truncated:
                ep["outcome"] = "survived"
                done[i] = True

    for ep in episodes:
        ep.setdefault("outcome", "survived")
        ep["return"] = round(sum(f.get("r", 0.0) for f in ep["frames"]), 2)
    return episodes


def write_replay_html(
    checkpoints: dict[str, list[dict]],
    path: str | Path,
    title: str = "Gridworld Replays",
    subtitle: str = "",
) -> Path:
    """Writes the replay page. `checkpoints` maps a label (e.g. "untrained", "after 100k steps") to the episodes
    recorded at that point; the page lets you switch between them."""
    data = {
        "title": title,
        "subtitle": subtitle,
        "tiles": {t.name.lower(): int(t) for t in Tile},
        "checkpoints": [{"label": label, "episodes": eps} for label, eps in checkpoints.items()],
    }
    template = TEMPLATE_PATH.read_text()
    html = template.replace("__TITLE__", title).replace("__DATA__", json.dumps(data, separators=(",", ":")))
    path = Path(path)
    path.write_text(html)
    return path
