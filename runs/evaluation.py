"""
How well a policy plays: one full episode on each of a fixed set of layouts, with the results averaged over them. Used
during training (train_gridworld.py) and by the experiment analyses (analysis/).
"""

from dataclasses import replace
from typing import Callable

import torch as t

from envs.gridworld.batched_env import BatchedGridworldEnv
from envs.gridworld.config import GridworldConfig
from envs.gridworld.observation import CHANNEL_INDEX

EVAL_ENVS, EVAL_SEED = 64, 12345  # the same 64 layouts at every evaluation


def metric_labels(config: GridworldConfig) -> dict[str, str]:
    """The evaluation results, in the order they're reported, with their labels for the viewer's charts."""
    labels = {
        "survival": f"Survived all {config.max_steps} steps (fraction)",
        "deaths_starvation": "Died of starvation (fraction of episodes)",
        "deaths_fire": "Died in fire (fraction of episodes)",
        "return": "Episode return",
        "fire_steps": "Steps on fire per episode",
        "food_eaten": "Food eaten per episode",
        "pushes": "Pushes of the NPC per episode",
        "npc_fire_steps": "NPC steps on fire per episode",
        "npc_survival": "NPC alive when the episode ended (fraction)",
    }
    if not config.npc_enabled:
        del labels["npc_survival"]
    return labels


@t.inference_mode()
def evaluate(
    act_fn: Callable[[t.Tensor], t.Tensor],
    config: GridworldConfig,
    device: t.device,
    num_layouts: int = EVAL_ENVS,
) -> dict[str, float]:
    """
    Runs one full episode on each of `num_layouts` fixed layouts and averages over them. Returns the metrics named in
    `metric_labels`. NPC survival is whether the NPC is still alive when the agent's episode ends (after max_steps, or
    when the agent dies). A death counts as "fire" if the agent died standing on fire and as "starvation" otherwise
    (the only other way to lose health).
    """
    agent_energy = lambda obs: (obs[:, CHANNEL_INDEX["energy"]] * obs[:, CHANNEL_INDEX["self"]]).sum((-1, -2))
    # Evaluation episodes run to the end even when the NPC is hurt: dropping is only for what the agent trains on
    env = BatchedGridworldEnv(num_layouts, EVAL_SEED, device, replace(config, drop_npc_harm=False))
    obs, _ = env.reset()
    alive = t.ones(num_layouts, dtype=t.bool, device=obs.device)
    npc_alive = alive.clone()
    episode_return, fire_steps, food_eaten, pushes, npc_fire_steps, deaths_fire, deaths_starvation = (
        t.zeros(num_layouts, device=obs.device) for _ in range(7)
    )
    for _ in range(config.max_steps):
        next_obs, reward, terminated, truncated, infos = env.step(act_fn(obs))
        episode_return += reward * alive
        fire_steps += infos["agent_on_hazard"].float() * alive
        pushes += infos["npc_pushed"].float() * alive
        npc_fire_steps += infos["npc_on_hazard"].float() * alive
        ate = agent_energy(next_obs) > agent_energy(obs)  # energy only goes up by eating
        food_eaten += (ate & ~terminated & alive).float()
        npc_alive &= ~(infos["npc_died"] & alive)
        died_now = terminated & alive
        deaths_fire += (died_now & infos["agent_on_hazard"]).float()
        deaths_starvation += (died_now & ~infos["agent_on_hazard"]).float()
        alive &= ~terminated  # count each env's first episode only
        obs = next_obs
    results = {
        "survival": alive.float().mean().item(),
        "deaths_starvation": deaths_starvation.mean().item(),
        "deaths_fire": deaths_fire.mean().item(),
        "return": episode_return.mean().item(),
        "fire_steps": fire_steps.mean().item(),
        "food_eaten": food_eaten.mean().item(),
        "pushes": pushes.mean().item(),
        "npc_fire_steps": npc_fire_steps.mean().item(),
        "npc_survival": npc_alive.float().mean().item(),
    }
    return {key: results[key] for key in metric_labels(config)}
