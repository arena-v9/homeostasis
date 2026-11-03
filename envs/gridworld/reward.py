"""
The agent's reward for one step, by agent model (`GridworldConfig.agent_model`):

  homeostatic   The reduction in its own drive (drive.py, Keramati & Gutkin 2014): rewarded for eating, punished for
                getting hungry, hurt or killed.
  psychopath    `food_reward` for each food item it eats, and nothing else. Its body can't be hurt (neither fire nor
                starvation lowers its health, see dynamics.py), so it never dies and only maximises the food it eats.

Neither model's reward includes the NPC.
"""

from envs.gridworld.config import GridworldConfig
from envs.gridworld.drive import drive_reduction_reward
from envs.gridworld.state import GridworldState


def step_reward(state: GridworldState, next_state: GridworldState, agent_ate: bool, config: GridworldConfig) -> float:
    """The agent's reward for the step from `state` to `next_state`, in which it ate if `agent_ate`."""
    if config.agent_model == "psychopath":
        return config.food_reward * agent_ate
    return drive_reduction_reward(state, next_state, config)
