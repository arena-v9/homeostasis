"""
Training diagnostics for catastrophic forgetting, following ARENA's DQN chapter
(https://learn.arena.education/chapter2_rl/21_dqn/3-main-dqn-algorithm/). train_gridworld.py logs them at every
evaluation, next to the evaluation results (whose survival rate is ARENA's "how long the greedy policy survives"):

  buffer_terminal_share  Share of the replay buffer's transitions that end in death. If it falls towards zero, the
                         agent is no longer training on any failures and may forget what they look like.
  td_loss                The TD loss, the quantity training minimises, on a large sample from the buffer.
  action_gap             max_a Q(s, a) - min_a Q(s, a), averaged over a fixed set of states from random play. A
                         shrinking gap means the network has stopped telling the actions apart.
  doomed_q_error         For a fixed set of doomed states (every action dies on the next step), the network's
                         max_a Q(s, a) minus the true value. In ARENA's CartPole the true value is 0; here the step
                         that kills the agent is rewarded reward_scale * (D(s) - D_max) whatever the action, so that's
                         the true value. A rising error means the network has started treating doomed states as
                         survivable.

The fixed state sets are built once per run from a fixed seed, and the buffer is sampled with the diagnostics' own
random generator, so logging them doesn't change training.
"""

from dataclasses import dataclass

import numpy as np
import torch as t

from envs.gridworld.config import GridworldConfig
from envs.gridworld.dynamics import ACTION_OFFSETS, Action, offset_position
from envs.gridworld.episode import new_episode, observe_episode, step_episode
from envs.gridworld.state import GridworldState

PROBE_SEED = 777
NUM_PROBE_STATES = 1024
TD_LOSS_SAMPLES = 10_000

LABELS = {
    "buffer_terminal_share": "Buffer: share of transitions that are deaths",
    "td_loss": f"TD loss on {TD_LOSS_SAMPLES:,} buffer transitions",
    "action_gap": "Action gap (max Q − min Q) on random-play states",
    "doomed_q_error": "Doomed states: max Q minus true value",
}


@dataclass
class Probes:
    """The fixed state sets the diagnostics evaluate the Q-network on."""

    random_play_obs: t.Tensor  # (NUM_PROBE_STATES, channels, size, size)
    doomed_obs: t.Tensor  # (NUM_PROBE_STATES, channels, size, size)
    doomed_true_q: t.Tensor  # (NUM_PROBE_STATES,): the reward of dying from each doomed state


def build_probes(config: GridworldConfig, device: t.device, seed: int = PROBE_SEED) -> Probes:
    rng = np.random.default_rng(seed)
    random_play = random_play_states(config, NUM_PROBE_STATES, rng)
    doomed, true_q = doomed_states(config, NUM_PROBE_STATES, rng)
    to_tensor = lambda array: t.from_numpy(np.stack(array)).float().to(device)
    return Probes(to_tensor(random_play), to_tensor(doomed), to_tensor(true_q))


def random_play_states(config: GridworldConfig, num_states: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Observations from episodes played with random actions (and the scripted NPC), one per step."""
    observations = []
    while len(observations) < num_states:
        episode = new_episode(config, rng)
        for _ in range(config.max_steps):
            observations.append(observe_episode(episode, config))
            episode, result = step_episode(episode, Action(rng.integers(len(Action))), config, rng)
            if result.terminated or len(observations) == num_states:
                break
    return observations


def doomed_states(
    config: GridworldConfig, num_states: int, rng: np.random.Generator
) -> tuple[list[np.ndarray], list[float]]:
    """
    States where the agent starves to death on the next step whatever it does: energy 0, health no more than one step
    of starvation damage, and no food on or next to its square. Each candidate is checked by trying every action.
    Returns their observations and the true value of each (the reward of the dying step, the same for every action).
    """
    observations, true_values = [], []
    while len(observations) < num_states:
        episode = new_episode(config, rng)
        episode.state.agent.energy = 0.0
        episode.state.agent.health = float(rng.uniform(0.1, 1.0) * config.starvation_damage)
        _remove_food_within_reach(episode.state)
        outcomes = [step_episode(episode, action, config, rng)[1] for action in Action]
        if all(outcome.terminated for outcome in outcomes):
            rewards = [outcome.reward for outcome in outcomes]
            assert max(rewards) - min(rewards) < 1e-9, "dying should be rewarded the same whatever the action"
            observations.append(observe_episode(episode, config))
            true_values.append(rewards[0])
    return observations, true_values


def _remove_food_within_reach(state: GridworldState) -> None:
    for offset in ACTION_OFFSETS.values():
        state.food[offset_position(state.agent.position, offset)] = False


@t.inference_mode()
def diagnose(trainer, probes: Probes, generator: t.Generator) -> dict[str, float]:
    """The diagnostics in LABELS, for the trainer's current Q-network and replay buffer."""
    sample = trainer.buffer.sample(TD_LOSS_SAMPLES, generator)
    td_loss = trainer.td_loss(sample)
    q_values = trainer.q_network(probes.random_play_obs)
    doomed_max_q = trainer.q_network(probes.doomed_obs).max(-1).values
    return {
        "buffer_terminal_share": trainer.buffer.terminated.float().mean().item(),
        "td_loss": td_loss.item(),
        "action_gap": (q_values.max(-1).values - q_values.min(-1).values).mean().item(),
        "doomed_q_error": (doomed_max_q - probes.doomed_true_q).mean().item(),
    }
