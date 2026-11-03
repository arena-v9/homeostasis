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

    With `harm_head`, a second output reads the convolutions' features at every square through one 1x1 convolution:
    the predicted health change of a body standing there (`harm`), the auxiliary task in DQNTrainer.harm_loss. The same
    weights read every square, so the head asks the same question of every body, the agent's and the others'.
    """

    layers: nn.Sequential

    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        num_actions: int,
        channels: int = 32,
        hidden: int = 128,
        harm_head: bool = False,
    ):
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
        self.harm_head = nn.Conv2d(channels, 1, kernel_size=1) if harm_head else None

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)

    def harm(self, x: Tensor) -> Tensor:
        """The predicted health change of a body on each square, shape (batch, height, width)."""
        features = self.layers[:4](x)  # both convolutions and their ReLUs
        return self.harm_head(features).squeeze(1)
