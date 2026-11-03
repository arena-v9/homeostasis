"""
Tests for the TD loss (DQNTrainer.td_loss), which uses Double DQN's target.

Run from the repo root: python -m pytest tests/test_trainer.py
"""

from types import SimpleNamespace

import torch as t

from dqn.networks import ConvQNetwork
from dqn.replay_buffer import ReplayBufferSamples
from dqn.trainer import DQNTrainer


class FixedQ(t.nn.Module):
    """A 'network' that gives the same Q-values for every observation."""

    def __init__(self, q_values: list[float]):
        super().__init__()
        self.q_values = t.tensor(q_values)

    def forward(self, obs: t.Tensor) -> t.Tensor:
        return self.q_values.expand(len(obs), -1).clone()


def td_target() -> float:
    """The TD target for one non-terminal transition with reward 0 and gamma 1, whose predicted Q-value is 0."""
    trainer = DQNTrainer.__new__(DQNTrainer)  # just the parts td_loss uses
    trainer.args = SimpleNamespace(gamma=1.0)
    trainer.q_network = FixedQ([0.0, 5.0, 1.0])  # prefers action 1
    trainer.target_network = FixedQ([9.0, 2.0, 3.0])  # values action 1 at 2, but its own best (action 0) at 9
    data = ReplayBufferSamples(obs=t.zeros(1, 1), actions=t.tensor([0]), rewards=t.tensor([0.0]),
                               terminated=t.tensor([False]), next_obs=t.zeros(1, 1))
    return trainer.td_loss(data).item() + 0.5  # Huber loss of an error x > 1 is x - 0.5


def test_the_q_network_chooses_the_next_action_and_the_target_network_values_it():
    assert td_target() == 2.0  # plain DQN would use the target network's own best value, 9


def test_the_harm_loss_only_counts_squares_with_a_body():
    trainer = DQNTrainer.__new__(DQNTrainer)
    trainer.q_network = ConvQNetwork((9, 3, 3), 5, harm_head=True)
    t.nn.init.zeros_(trainer.q_network.harm_head.weight)
    t.nn.init.zeros_(trainer.q_network.harm_head.bias)  # predicts 0 everywhere
    harm = t.full((1, 3, 3), float("nan"))
    harm[0, 1, 1] = -0.3
    data = ReplayBufferSamples(obs=t.zeros(1, 9, 3, 3), actions=t.tensor([0]), rewards=t.tensor([0.0]),
                               terminated=t.tensor([False]), next_obs=t.zeros(1, 9, 3, 3), harm=harm)
    assert abs(trainer.harm_loss(data).item() - 0.09) < 1e-6


def test_networks_without_the_head_load_weights_saved_before_it_existed():
    saved = ConvQNetwork((9, 7, 14), 5).state_dict()
    assert "harm_head.weight" not in saved
    ConvQNetwork((9, 7, 14), 5).load_state_dict(saved)
    assert ConvQNetwork((9, 7, 14), 5, harm_head=True).harm(t.zeros(2, 9, 7, 14)).shape == (2, 7, 14)
