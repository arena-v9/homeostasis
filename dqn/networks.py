"""
The Q-networks. `DQNTrainer` picks one from the observation shape: QNetwork (an MLP) for flat vector observations
(the probe environments), ConvQNetwork (a CNN) for (channels, height, width) observations (the gridworld).
"""

import torch.nn as nn
from torch import Tensor


class QNetwork(nn.Module):
    """
    MLP for flat vector observations: Linear -> ReLU -> Linear -> ReLU -> Linear, with the layers in an
    `nn.Sequential` called `layers` (the tests check this).
    """

    layers: nn.Sequential

    def __init__(self, obs_shape: tuple[int], num_actions: int, hidden_sizes: list[int] = [120, 84]):
        super().__init__()
        assert len(obs_shape) == 1, "Expecting a single vector of observations"
        in_features_list = [obs_shape[0]] + hidden_sizes
        out_features_list = hidden_sizes + [num_actions]
        layers = []
        for i, (in_features, out_features) in enumerate(zip(in_features_list, out_features_list)):
            layers.append(nn.Linear(in_features, out_features))
            if i < len(in_features_list) - 1:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class ConvQNetwork(nn.Module):
    """
    The Q-network for the gridworld: two 3x3 convolutions (padded, so the map stays size x size), then two
    linear layers. For the 9 x 7 x 7 observation:
        Conv 3x3, 32 filters, ReLU -> Conv 3x3, 32 filters, ReLU -> Flatten (32 * 7 * 7 = 1568)
        -> Linear 1568 -> 128, ReLU -> Linear 128 -> num_actions
    The convolutions apply the same weights at every square, so a feature learned from the agent's own body also
    responds to the NPC's body on the shared channels.
    """

    layers: nn.Sequential

    def __init__(self, obs_shape: tuple[int, int, int], num_actions: int, channels: int = 32, hidden: int = 128):
        super().__init__()
        assert len(obs_shape) == 3, "Expecting observations of shape (channels, height, width)"
        in_channels, height, width = obs_shape
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(channels * height * width, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)
