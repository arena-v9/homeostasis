from torch import Tensor

from common import Pendulum
from gpu_env import angle_normalize


def pendulum_reward(theta: Tensor, theta_dot: Tensor, torque: Tensor) -> Tensor:
    """
    Reward for the Pendulum swing-up task. All arguments are (num_envs,) tensors:
        theta     : pole angle in radians; theta=0 is straight up, theta=±π is hanging down (wraps).
        theta_dot : angular velocity, in [-8, 8].
        torque    : the applied action, in [-2, 2].
    Returns a (num_envs,) reward tensor, larger when the pole is upright and still.
    """
    # One good choice (the canonical gym reward): a negative quadratic cost that is zero only when the
    # pole is exactly upright, motionless, and no torque is applied. angle_normalize(theta) is the
    # signed angular distance from upright, so its square penalises being away from the top; the other
    # two terms gently discourage spinning and wasting control effort.
    return -(angle_normalize(theta) ** 2 + 0.1 * theta_dot ** 2 + 0.001 * torque ** 2)


def _pendulum_reward_method(self, action):
    u = action.reshape(self.num_envs).clamp(-self.max_torque, self.max_torque)
    return pendulum_reward(self.state[:, 0], self.state[:, 1], u)


class MyPendulum(Pendulum):
    def reward_function(self, action):
        return _pendulum_reward_method(self, action)
