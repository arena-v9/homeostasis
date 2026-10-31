import torch as t
import torch.nn as nn
from torch import Tensor

from common import layer_init


class Actor(nn.Module):
    actor_mu: nn.Sequential
    actor_log_sigma: nn.Parameter

    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.actor_mu = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )
        self.actor_log_sigma = nn.Parameter(t.zeros(1, action_dim))

    def forward(self, obs) -> tuple[Tensor, Tensor, t.distributions.Normal]:
        mu = self.actor_mu(obs)
        sigma = t.exp(self.actor_log_sigma).broadcast_to(mu.shape)
        dist = t.distributions.Normal(mu, sigma)
        return mu, sigma, dist
