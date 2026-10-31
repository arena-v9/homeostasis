import torch.nn as nn
from torch import Tensor

from common import layer_init


class Critic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def forward(self, obs) -> Tensor:
        value = self.critic(obs)
        return value
