from torch import Tensor

from common import CartPole


def spin_cart_reward_function(self, action: Tensor) -> Tensor:
    """Reward for the spinning-cart task. Like `cartpole_reward_function`, this is called by `step`
    after the physics update (`self.state` is the new (num_envs, 4) tensor [x, v, theta, omega])
    and should return a (num_envs,) reward tensor."""
    x, v, theta, omega = self.state.unbind(-1)  # each (num_envs,): position, velocity, angle, angular velocity
    # Reward function incentivises fast spinning while staying still & near centre
    rotation_speed_reward = (0.1 * omega.abs()).clamp(max=1.0)
    stability_penalty = ((x / 2.5).abs() + (v / 10).abs()).clamp(min=1.0)
    return rotation_speed_reward - 0.5 * stability_penalty


class SpinCart(CartPole):
    def reward_function(self, action):
        return spin_cart_reward_function(self, action)

    def terminated(self):
        # Allow full 360-degree rotation: unlike the parent class, only terminate when the cart
        # leaves the track (the pole angle no longer ends the episode).
        x = self.state[:, 0]
        return x.abs() > self.x_threshold
