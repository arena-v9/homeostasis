"""
Batched, tensor-native CartPole (adapted from ARENA's `chapter2_rl/exercises/gpu_env.py`, originally from
https://gist.github.com/ngoodger/6cf50a05c9b3c189be30fab34ab5d85c). Classic cart-pole system implemented by
Rich Sutton et al., http://incompleteideas.net/sutton/book/code/pole.c (permalink: https://perma.cc/C9ZM-652R).
"""

import math

import numpy as np
import torch
from gymnasium import spaces

from common import device as default_device
from gpu_env import GPUEnv, track_episode_stats


class CartPole(GPUEnv):
    """
    A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum
    starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.

    Observation: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf
    Actions: Discrete(2)
        0     Push cart to the left
        1     Push cart to the right
    Reward:
        1 for every step the pole stays up, and 0 on the step that terminates the episode
        (unlike Gym's CartPole, which also gives 1 on the terminating step)
    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]
    Episode Termination:
        Pole Angle is more than 12 degrees, or Cart Position is more than 2.4 (terminated);
        episode length reaches 500 (truncated).
    """

    def __init__(self, env_id=None, num_envs=1, seed=0, device=None):
        # Uniform env signature (env_id, num_envs, seed, device). CartPole IS the env, so env_id is ignored.
        device = device if device is not None else default_device
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
        self.num_envs = num_envs

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds.
        high = np.array(
            [self.x_threshold * 2, np.finfo(np.float32).max, self.theta_threshold_radians * 2, np.finfo(np.float32).max],
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        # Vector-env style `single_*` aliases, as used by the trainer
        self.single_action_space = self.action_space
        self.single_observation_space = self.observation_space

        # _terminated / _truncated are the latest per-env flags. Start True so the first reset() initialises
        # every env.
        self._terminated = torch.full([num_envs], True, dtype=torch.bool, device=device)
        self._truncated = torch.full([num_envs], True, dtype=torch.bool, device=device)
        self.state = torch.zeros([self.num_envs, 4], dtype=torch.float32, device=device)
        self.timestep = torch.zeros([self.num_envs], dtype=torch.int, device=device)

        self.MAX_LENGTH = 500

        self.device = device
        # Seeded generator so resets are reproducible
        self.gen = torch.Generator(device=device).manual_seed(seed)

    def terminated(self):
        """A CartPole env terminates when the pole falls past +/-12 degrees or the cart leaves the
        +/-x_threshold track. Computed from the current state."""
        x, theta = self.state[:, 0], self.state[:, 2]
        return (
            (x < -self.x_threshold)
            | (x > self.x_threshold)
            | (theta < -self.theta_threshold_radians)
            | (theta > self.theta_threshold_radians)
        )

    def reward_function(self, action):
        """Default CartPole reward: +1 for every step the pole is still up, 0 on the terminal step (float).
        Uses the termination flags computed for the current step in `step`."""
        return (~self._terminated).float()

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state[:, 0], self.state[:, 1], self.state[:, 2], self.state[:, 3]
        force = self.force_mag * ((action * 2.0) - 1.0)

        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)

        # For the interested reader: https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        # Build a NEW state tensor rather than writing into self.state in place: callers hold a reference to the
        # observation we returned last step, and an in-place update would silently mutate it.
        self.state = torch.stack((x, x_dot, theta, theta_dot), dim=1)

        self._terminated = self.terminated()
        self.timestep += 1
        self._truncated = self.timestep == self.MAX_LENGTH
        reward = self.reward_function(action)

        self.state, infos = self.reset()
        infos.update(track_episode_stats(self, reward, self._terminated, self._truncated))
        return self.state, reward, self._terminated, self._truncated, infos

    def reset(self):
        done = self._truncated | self._terminated
        new_state = torch.where(
            done.unsqueeze(1),
            (torch.rand(self.num_envs, 4, generator=self.gen, device=self.device) - 0.5) / 10.0,
            self.state,
        )
        infos = {"final_observation": self.state.clone()}  # the state before any auto-reset
        self.state = new_state
        self.timestep[done] = 0
        return self.state, infos

    def draw(self, o, canvas, ox, oy, w, h):
        """Grid renderer: the cart as a black box on a rail, with the pole drawn from the cart at its
        current angle. `o` is [x, x_dot, theta, theta_dot]."""
        import cv2

        x, theta = float(o[0]), float(o[2])
        scale = w / (2 * self.x_threshold * 1.1)
        polelen = int(scale)
        cw, ch = max(int(scale * 0.5), 6), 8
        cy = oy + int(h * 0.6)
        cv2.line(canvas, (ox, cy), (ox + w, cy), (200, 200, 200), 1)
        cx = min(max(int(ox + w / 2 + x * scale), ox + cw), ox + w - cw)
        cv2.rectangle(canvas, (cx - cw, cy - ch), (cx + cw, cy + ch), (0, 0, 0), -1)
        cv2.line(
            canvas,
            (cx, cy),
            (int(cx + polelen * math.sin(theta)), int(cy - polelen * math.cos(theta))),
            (60, 80, 200),
            3,
        )
