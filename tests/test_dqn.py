"""
Tests for the DQN pieces (adapted from ARENA's `chapter2_rl/exercises/part21_dqn/tests.py` and the 2.2.1 notebook):
unit tests for the Q-network, exploration schedule, epsilon-greedy policy and agent, then a short DQN training run
on each of the five probe environments. ARENA's versions compare against its reference `solutions.py`; these check
against independently computed expectations instead.

Run from the repo root: python -m pytest tests/test_dqn.py
"""

import numpy as np
import pytest
import torch as t
import torch.nn as nn
from gymnasium.spaces import Box, Discrete

from common import device
from dqn.agent import DQNAgent
from dqn.args import DQNArgs
from dqn.exploration import epsilon_greedy_policy, linear_schedule
from dqn.networks import QNetwork
from dqn.replay_buffer import ReplayBuffer
from dqn.trainer import DQNTrainer
from envs.probe_envs import Probe1, Probe2, Probe3, Probe4, Probe5


class CountingEnv:
    """
    Tiny batched env for testing what the agent stores in the replay buffer. Each env's observation is its step
    count within the episode, [0], [1], ..., and every episode ends when the count reaches EPISODE_LENGTH: by
    *termination* for even-numbered envs and by *truncation* (a time limit) for odd-numbered ones. Done envs
    auto-reset to [0] inside `step`, reporting the true final observation [EPISODE_LENGTH] in
    `infos["final_observation"]`. So for every correctly stored transition, next_obs == obs + 1.
    """

    EPISODE_LENGTH = 5

    def __init__(self, num_envs: int, device: t.device = device):
        self.num_envs = num_envs
        self.device = device
        self.single_observation_space = Box(np.array([0.0], dtype=np.float32), np.array([self.EPISODE_LENGTH], dtype=np.float32))
        self.single_action_space = Discrete(2)  # actions are ignored
        self.ends_by_termination = t.arange(num_envs, device=device) % 2 == 0
        self.state = t.zeros([num_envs, 1], dtype=t.float32, device=device)

    def reset(self):
        self.state = t.zeros([self.num_envs, 1], dtype=t.float32, device=self.device)
        return self.state.clone(), {}

    def step(self, action):
        self.state = self.state + 1
        ended = self.state[:, 0] == self.EPISODE_LENGTH
        terminated = ended & self.ends_by_termination
        truncated = ended & ~self.ends_by_termination
        reward = t.ones(self.num_envs, dtype=t.float32, device=self.device)
        infos = {"final_observation": self.state.clone()}
        self.state = t.where(ended.unsqueeze(-1), t.zeros_like(self.state), self.state)  # auto-reset
        return self.state.clone(), reward, terminated, truncated, infos


def get_steps(obj):
    """
    Gets steps, in both the case where it's named "step" and "steps" (for consistency with the wandb.log
    argument `step`, the agent should use `step`).
    """
    if hasattr(obj, "step"):
        return obj.step
    elif hasattr(obj, "steps"):
        raise Exception(
            f"Object {obj} has a `steps` attribute, but not a `step` attribute. Please rename `steps` to `step` (for consistency with the wandb.log argument `step`)."
        )
    else:
        raise Exception(f"Object {obj} has neither a `step` nor a `steps` attribute.")


def test_linear_schedule():
    # start_e=1.0 decays linearly to end_e=0.05 over the first 0.5 * 500 = 250 steps, then stays at 0.05
    kwargs = dict(start_e=1.0, end_e=0.05, exploration_fraction=0.5, total_timesteps=500)
    expected = t.tensor([max(1.0 - 0.95 * step / 250, 0.05) for step in range(500)])
    actual = t.tensor([linear_schedule(step, **kwargs) for step in range(500)])
    assert expected.shape == actual.shape
    t.testing.assert_close(actual, expected, check_dtype=False)
    assert linear_schedule(0, **kwargs) == 1.0, "Epsilon should equal start_e at step 0."
    assert abs(linear_schedule(125, **kwargs) - 0.525) < 1e-9, "Epsilon should be halfway between start_e and end_e halfway through exploration."
    assert abs(linear_schedule(10_000, **kwargs) - 0.05) < 1e-9, "Epsilon should stay at end_e after exploration ends."


def test_epsilon_greedy_policy():

    # Lots of envs, so that "what fraction of envs did X" is a stable statistic
    num_envs = 2000
    envs = Probe5(num_envs=num_envs, seed=0, device=device)  # 2 actions, random 0/1 observations
    obs, _ = envs.reset()
    num_actions = envs.single_action_space.n
    q_network = QNetwork(envs.single_observation_space.shape, num_actions).to(device)
    rng = t.Generator(device=device).manual_seed(0)

    with t.inference_mode():
        greedy_actions = q_network(obs).argmax(-1)

    def get_actions(epsilon):
        actions = epsilon_greedy_policy(envs, q_network, rng, obs, epsilon)
        assert isinstance(actions, t.Tensor), f"Expected actions to be a tensor, got {type(actions)}."
        assert actions.shape == (num_envs,), f"Expected actions of shape ({num_envs},), got {actions.shape}."
        assert not actions.is_floating_point(), f"Expected an integer dtype for actions, got {actions.dtype}."
        assert actions.device.type == obs.device.type, "Actions should be on the same device as the observations."
        assert ((actions >= 0) & (actions < num_actions)).all(), "Some actions are outside [0, num_actions)."
        return actions

    def fraction_greedy(epsilon):
        """Fraction of envs whose sampled action equals the greedy action."""
        return (get_actions(epsilon) == greedy_actions).float().mean().item()

    # epsilon = 0: every env acts greedily
    assert t.equal(get_actions(0.0), greedy_actions), "With epsilon=0 every action should be the greedy (argmax Q) action."

    # epsilon = 1: every env acts randomly, so it agrees with greedy 1/num_actions = 50% of the time
    frac = fraction_greedy(1.0)
    assert 0.4 < frac < 0.6, f"With epsilon=1 (uniformly random actions) you should agree with greedy ~50% of the time, got {frac:.2f}."

    # 0 < epsilon < 1: each env explores INDEPENDENTLY with probability epsilon, so on EVERY call the fraction acting
    # greedily should be close to (1 - epsilon) + epsilon / num_actions. If you flipped one coin for all envs instead,
    # each call would give either ~1.0 (nobody explored) or ~0.5 (everybody did) - the same average, but a huge spread.
    for epsilon in [0.1, 0.5]:
        expected = (1 - epsilon) + epsilon / num_actions
        fracs = [fraction_greedy(epsilon) for _ in range(10)]
        assert all(abs(frac - expected) < 0.08 for frac in fracs), f"""
With epsilon={epsilon}, the fraction of envs taking the greedy action should be close to {expected:.2f} on every call
(each env explores independently with probability epsilon). Over 10 calls yours ranged from {min(fracs):.2f} to
{max(fracs):.2f}. If it jumps between ~1.0 and ~{1 / num_actions:.2f}, you are flipping a single coin for all envs
rather than one per env."""



def test_agent():
    # Set up args and envs (4 counting envs: envs 0 and 2 end by termination, envs 1 and 3 by truncation)
    args = DQNArgs(buffer_size=100)
    envs = CountingEnv(num_envs=4)
    num_envs = envs.num_envs
    action_shape = envs.single_action_space.shape
    obs_shape = envs.single_observation_space.shape
    num_actions = envs.single_action_space.n
    rng = t.Generator(device=device).manual_seed(args.seed)
    L = CountingEnv.EPISODE_LENGTH

    # Set up networks, buffer, and agent
    q_network = QNetwork(obs_shape, num_actions).to(device)
    buffer = ReplayBuffer(num_envs, obs_shape, action_shape, args.buffer_size, args.seed, device)
    agent = DQNAgent(
        envs,
        buffer,
        q_network,
        args.start_e,
        args.end_e,
        args.exploration_fraction,
        args.total_timesteps,
        rng,
    )

    # Run a bunch of steps, to fill out the replay buffer
    n_steps = 256
    for i in range(n_steps):
        infos = agent.play_step()
        assert isinstance(infos, dict), "`play_step` should return `infos` (the last return argument from `envs.step`)."

    # Check steps (each call to play_step advances every env once, i.e. num_envs environment steps)
    assert get_steps(agent) == n_steps * num_envs, (
        f"Agent did not take the expected number of steps: expected self.step={n_steps * num_envs}, got {get_steps(agent)}."
    )

    # Check shapes of everything in replay experiences
    for name in ["obs", "actions", "rewards", "terminated", "next_obs"]:
        data = getattr(buffer, name)
        expected_shape = (args.buffer_size, *obs_shape) if "obs" in name else (args.buffer_size,)
        assert isinstance(data, t.Tensor), f"Replay buffer's {name} should be a tensor, got {type(data)}."
        assert data.shape == expected_shape, f"Replay buffer's {name} has incorrect shape: expected {expected_shape}, got {data.shape}."

    # The buffer holds one row per env per step, in the order they were added. So row i comes from env
    # i % num_envs (n_steps * num_envs and buffer_size are both multiples of num_envs), and the SAME env one step
    # later is row i + num_envs.
    assert (n_steps * num_envs) % num_envs == 0 and args.buffer_size % num_envs == 0
    obs_rows = buffer.obs[:, 0].cpu()
    next_obs_rows = buffer.next_obs[:, 0].cpu()
    terminated_rows = buffer.terminated.cpu()

    for i, (obs, next_obs, terminated) in enumerate(zip(obs_rows.tolist(), next_obs_rows.tolist(), terminated_rows.tolist())):
        env_idx = i % num_envs
        episode_ended = next_obs == L

        # 'obs' is never the final observation (envs auto-reset), so it should always be in [0, L)
        assert 0 <= obs < L, f"""
Agent observation 'obs' was {obs} at row {i}, but it should always be in [0, {L}).
Did you accidentally edit 'obs' before adding it to the buffer, when you meant to only change 'next_obs'?"""

        # 'next_obs' should always be the observation one step after 'obs' - including at the end of an episode,
        # where it's the true final observation, not the reset observation of the next episode.
        assert next_obs == obs + 1, f"""
For each (obs, actions, rewards, terminated, next_obs) added to the buffer, 'next_obs' should be the observation one
step after 'obs'. But you have obs={obs}, next_obs={next_obs} at row {i}.
If next_obs is 0, you've probably used the reset obs from the new episode (which is what 'envs.step' returns for
envs that just finished). Instead you need to use the true final obs, from infos["final_observation"]."""

        # 'terminated' should be True only for genuine terminations, not for truncations (time limits)
        expected_terminated = episode_ended and env_idx % 2 == 0
        assert terminated == expected_terminated, f"""
Row {i} (env {env_idx}, next_obs={next_obs}) has terminated={terminated}, expected {expected_terminated}.
Envs 0 and 2 end by termination and envs 1 and 3 by truncation. The buffer should store 'terminated', not
'terminated | truncated': a truncated episode didn't really end, so its value should still be bootstrapped."""

        # Within an episode, 'next_obs' should match the 'obs' this env sees on its next step (num_envs rows later)
        if not episode_ended and i + num_envs < len(obs_rows):
            obs_next_step = obs_rows[i + num_envs].item()
            assert obs_next_step == next_obs, f"""
'next_obs' should be the same as the next value of 'obs' for that env - which is num_envs={num_envs} rows further on
in the buffer. But you have obs[i+num_envs] = {obs_next_step}, next_obs[i] = {next_obs} for {i=}."""

    # Check whether epsilon is being updated correctly. Epsilon is computed at the START of each play_step from
    # self.step, and self.step then advances by num_envs - so the last value used was at step - num_envs.
    epsilon_expected = linear_schedule(
        get_steps(agent) - num_envs,
        args.start_e,
        args.end_e,
        args.exploration_fraction,
        args.total_timesteps,
    )
    assert agent.epsilon == epsilon_expected, f"""
Agent's epsilon value is incorrect: yours = {agent.epsilon:.6f}, expected = {epsilon_expected:.6f}.
Remember to update using 'self.epsilon = linear_schedule(self.step, ...)' in 'get_actions'.
The value of 'self.step' should be 0 the first time this is called."""



def test_q_network():
    # Linear-ReLU-Linear-ReLU-Linear, with no ReLU on the output
    net = QNetwork(obs_shape=(4,), num_actions=2)
    assert isinstance(getattr(net, "layers", None), nn.Sequential)
    assert [type(layer) for layer in net.layers] == [nn.Linear, nn.ReLU, nn.Linear, nn.ReLU, nn.Linear]
    assert sum(p.nelement() for p in net.parameters()) == 10934


def test_probe_shapes():
    env = Probe1(num_envs=4)
    assert env.single_observation_space.shape == (1,)
    assert env.single_action_space.shape == ()
    obs, _ = env.reset()
    assert obs.shape == (4, 1)


@pytest.mark.parametrize("probe_idx", range(1, 6))
def test_probe(probe_idx: int):
    """
    Tests a probe environment by training a network on it & verifying that the value functions are
    in the expected range.
    """
    # Train our network on this probe env.
    args = DQNArgs(
        total_timesteps=3000 if probe_idx <= 2 else 5000,
        learning_rate=0.001,
        buffer_size=500,
        trains_per_target_update=20,
        steps_per_update=10,
        exploration_fraction=0.2,
        end_e=0.1,
    )
    probe_cls = [Probe1, Probe2, Probe3, Probe4, Probe5][probe_idx - 1]
    trainer = DQNTrainer(args, probe_cls(num_envs=1, seed=args.seed, device=args.device))
    trainer.train()

    # Get the correct set of observations, and corresponding values we expect
    obs_for_probes = [[[0.0]], [[-1.0], [+1.0]], [[0.0], [1.0]], [[0.0]], [[0.0], [1.0]]]
    expected_value_for_probes = [
        [[1.0]],
        [[-1.0], [+1.0]],
        [[args.gamma], [1.0]],
        [[-1.0, 1.0]],
        [[1.0, -1.0], [-1.0, 1.0]],
    ]
    tolerances = [5e-4, 5e-4, 5e-4, 5e-4, 1e-3]
    obs = t.tensor(obs_for_probes[probe_idx - 1]).to(device)

    # Calculate the actual value, and verify it
    value = trainer.q_network(obs)
    # A bf16 network (e.g. under autocast) can't represent these targets to better than its eps
    atol = max(tolerances[probe_idx - 1], t.finfo(value.dtype).eps)
    expected_value = t.tensor(expected_value_for_probes[probe_idx - 1]).to(device)
    t.testing.assert_close(value.float(), expected_value, atol=atol, rtol=0)
