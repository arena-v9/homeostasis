from dataclasses import dataclass

import torch as t

from common import EnvType, device


@dataclass
class PPOArgs:
    # Basic / global
    seed: int = 1
    env_id: str = "CartPole-v1"
    mode: EnvType = "classic-control"
    device: str = device  # envs, networks and replay memory all live here; "cpu" is fastest for CartPole

    # Wandb / logging
    use_wandb: bool = False
    video_log_freq: int | None = 50  # every N phases, render a 4x4 grid video of the latest rollout
                                     # (logged to wandb if use_wandb, shown inline if live_viz)
    live_viz: bool = True  # display the 4x4 grid video inline every `video_log_freq` phases (notebook only)
    overwrite_video: bool = True  # each live video replaces the previous one in the notebook output (False appends them)
    wandb_project_name: str = "PPOCartPole"
    wandb_entity: str | None = None

    # Duration of training. One phase = a rollout of num_envs * num_steps_per_rollout env steps, then
    # num_minibatches * batches_per_learning_phase gradient steps on it. CartPole is solved after ~20 phases
    # (~2 s on CPU, ~4 s on a GPU); more parallel envs make each phase's batch bigger but don't help here.
    total_phases: int = 100
    num_envs: int = 64
    num_steps_per_rollout: int = 64
    num_minibatches: int = 4
    batches_per_learning_phase: int = 4

    # Optimization hyperparameters (a high LR converges in seconds on CartPole)
    lr: float = 5e-3
    max_grad_norm: float = 0.5

    # RL hyperparameters
    gamma: float = 0.99

    # PPO-specific hyperparameters
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 1.0

    def __post_init__(self):
        self.batch_size = self.num_steps_per_rollout * self.num_envs

        assert self.batch_size % self.num_minibatches == 0, "batch_size must be divisible by num_minibatches"
        self.minibatch_size = self.batch_size // self.num_minibatches
        self.total_timesteps = self.total_phases * self.batch_size
        self.total_training_steps = self.total_phases * self.batches_per_learning_phase * self.num_minibatches
        self.device = t.device(self.device)
