from dataclasses import dataclass

import torch as t

from common import device


@dataclass
class DQNArgs:
    # Basic / global
    seed: int = 1
    device: str = device

    # Wandb / logging
    use_wandb: bool = False
    wandb_project_name: str = "DQN"
    wandb_entity: str | None = None

    # Duration of different phases / buffer memory settings
    total_timesteps: int = 300_000  # total environment steps (summed over all envs), including buffer filling
    steps_per_update: int = 1  # calls to `agent.play_step` (each = num_envs environment steps) per optimization step
    trains_per_target_update: int = 50  # optimization steps between target network updates
    buffer_size: int = 10_000  # replay buffer capacity (environment steps used to fill it before training)

    # Optimization hparams
    batch_size: int = 512
    learning_rate: float = 5e-3

    # RL-specific
    gamma: float = 0.99
    exploration_fraction: float = 0.1
    start_e: float = 1.0
    end_e: float = 0.02

    def __post_init__(self):
        self.device = t.device(self.device)
