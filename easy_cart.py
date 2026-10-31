from torch import Tensor

from common import CartPole


def cartpole_reward_function(self, action: Tensor) -> Tensor:
    """Shaped reward for CartPole. Called by `step` just after the physics update, so `self.state`
    is the new (num_envs, 4) state tensor with columns [x, v, theta, omega]. Should return a
    (num_envs,) reward tensor (ideally bounded in [0, 1], for easier comparison with the unshaped
    env)."""
    x, v, theta, omega = self.state.unbind(-1)  # each (num_envs,): position, velocity, angle, angular velocity
    # First reward: angle should be close to zero
    reward_1 = 1 - (theta / 0.2095).abs()
    # Second reward: position should be close to the center
    reward_2 = 1 - (x / 2.4).abs()
    # Combine both rewards (keep it in the [0, 1] range)
    return (reward_1 + reward_2) / 2


class EasyCart(CartPole):
    def reward_function(self, action):
        return cartpole_reward_function(self, action)
