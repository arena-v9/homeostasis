from dataclasses import dataclass

import torch as t

from common import device


@dataclass
class DQNArgs:
    # Basic / global
    seed: int = 1
    env_id: str = "Probe1-v0"  # any id registered in env_registry.ENVS
    num_envs: int = 16  # number of parallel environments to run
    device: str = device

    # Wandb / logging
    use_wandb: bool = False
    wandb_project_name: str = "DQN"
    wandb_entity: str | None = None
    video_log_freq: int | None = None  # number of optimization steps between wandb video upload
    steps_per_live_video: int | None = 5_000  # number of optimization steps between live video display
    overwrite_video: bool = True  # older live videos are overwritten by newer ones if True. Otherwise, they are plotted one after the other.

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
        assert self.total_timesteps - self.buffer_size >= self.steps_per_update * self.num_envs
        # Every call to `agent.play_step` advances all `num_envs` environments by one step; we take one optimization
        # step per `steps_per_update` such calls, so this is the total number of optimization steps
        self.total_training_steps = (self.total_timesteps - self.buffer_size) // (self.steps_per_update * self.num_envs)
        self.device = t.device(self.device)
