import numpy as np
import torch.nn as nn

from actor import Actor
from common import ENV_DICT, BatchedEnvs, EnvType, device, layer_init
from critic import Critic


def get_actor_and_critic(
    envs: BatchedEnvs,
    mode: EnvType = "classic-control",
) -> tuple[nn.Module, nn.Module]:
    """
    Returns (actor, critic), the networks used for PPO, using one of 3 different architectures depending on `mode`.
    """
    assert mode in ENV_DICT

    # Observations are continuous: `obs_shape` is the shape of one observation (e.g. (4,) for CartPole,
    # (4, 84, 84) for Atari), and `obs_dim` is the number of floats in it once flattened (4 for CartPole).
    obs_shape = envs.single_observation_space.shape
    obs_dim = int(np.prod(obs_shape))
    action_space = envs.single_action_space

    if mode in ("classic-control", "mountain-car", "probe"):
        # mountain-car (Discrete(3)) and the probe envs are also discrete classic-control tasks, so
        # they reuse this network. Discrete actions: `num_actions = action_space.n` is how many there are.
        num_actions = action_space.n
        actor, critic = get_actor_and_critic_classic(obs_dim, num_actions)
    elif mode == "atari":
        num_actions = action_space.n
        actor, critic = get_actor_and_critic_atari(obs_shape, num_actions)
    elif mode in ("mujoco", "swing-up", "pendulum"):
        # swing-up (cart + double-pendulum) and pendulum are continuous-action tasks, so they reuse
        # the MuJoCo Gaussian actor/critic. Continuous actions: `action_dim` is the length of the action vector.
        action_dim = int(np.prod(action_space.shape))
        actor, critic = get_actor_and_critic_mujoco(obs_dim, action_dim)
    else:
        raise ValueError(f"Unknown mode {mode!r}")

    return actor.to(device), critic.to(device)


def get_actor_and_critic_classic(obs_dim: int, num_actions: int):
    """
    Returns (actor, critic) in the "classic-control" case.
    """
    actor = nn.Sequential(
        layer_init(nn.Linear(obs_dim, 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, num_actions), std=0.01),
    )
    critic = nn.Sequential(
        layer_init(nn.Linear(obs_dim, 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, 1), std=1.0),
    )
    return actor, critic


def get_actor_and_critic_atari(obs_shape: tuple[int, ...], num_actions: int) -> tuple[nn.Sequential, nn.Sequential]:
    """
    Returns (actor, critic) in the "atari" case.
    """
    assert obs_shape[-1] % 8 == 4

    L_after_convolutions = (obs_shape[-1] // 8) - 3
    in_features = 64 * L_after_convolutions * L_after_convolutions

    hidden = nn.Sequential(
        layer_init(nn.Conv2d(4, 32, 8, stride=4, padding=0)),
        nn.ReLU(),
        layer_init(nn.Conv2d(32, 64, 4, stride=2, padding=0)),
        nn.ReLU(),
        layer_init(nn.Conv2d(64, 64, 3, stride=1, padding=0)),
        nn.ReLU(),
        nn.Flatten(),
        layer_init(nn.Linear(in_features, 512)),
        nn.ReLU(),
    )

    actor = nn.Sequential(hidden, layer_init(nn.Linear(512, num_actions), std=0.01))
    critic = nn.Sequential(hidden, layer_init(nn.Linear(512, 1), std=1))

    return actor, critic


def get_actor_and_critic_mujoco(obs_dim: int, action_dim: int):
    """
    Returns (actor, critic) in the "mujoco" case.
    """
    return Actor(obs_dim, action_dim), Critic(obs_dim)
