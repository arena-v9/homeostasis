"""
The homeostatic drive and the reward computed from it, following Keramati & Gutkin (2014). See DESIGN.md.
"""

from envs.gridworld.config import GridworldConfig
from envs.gridworld.state import Body, GridworldState


def body_drive(body: Body, config: GridworldConfig) -> float:
    """
    How far a body is from its setpoints (health = energy = 1):

        D = (deficit_health ** n + deficit_energy ** n) ** (1 / m),   deficit = 1 - value

    with n = `drive_exponent` and m = `drive_root`. A dead body has the maximum drive.
    """
    if not body.alive:
        return config.max_drive
    health_deficit = 1.0 - body.health
    energy_deficit = 1.0 - body.energy
    return (health_deficit**config.drive_exponent + energy_deficit**config.drive_exponent) ** (1.0 / config.drive_root)


def drive_reduction_reward(state: GridworldState, next_state: GridworldState, config: GridworldConfig) -> float:
    """
    Reward for the step from `state` to `next_state`: reward_scale * (D_before - D_after), from the agent's own drive.
    Positive when the step brought the agent closer to its setpoints, negative when it moved away. A step that kills
    the agent jumps its drive to the maximum.
    """
    return config.reward_scale * (body_drive(state.agent, config) - body_drive(next_state.agent, config))
