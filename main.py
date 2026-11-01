"""
Runs the tests from the 2.2.1 DQN notebook: unit tests for the exploration schedule, epsilon-greedy policy and
agent, then a short DQN training run on each of the five probe environments.
"""

import torch as t
import torch.nn as nn

import tests
from common import device
from dqn_agent import DQNAgent
from dqn_args import DQNArgs
from dqn_trainer import DQNTrainer
from exploration import epsilon_greedy_policy, linear_schedule
from plotting import line
from probe_envs import Probe1
from q_network import QNetwork


def run_tests():
    net = QNetwork(obs_shape=(4,), num_actions=2)
    n_params = sum((p.nelement() for p in net.parameters()))
    assert isinstance(getattr(net, "layers", None), nn.Sequential)
    print(net)
    print(f"Total number of parameters: {n_params}")
    print("You should manually verify network is Linear-ReLU-Linear-ReLU-Linear")
    assert not isinstance(net.layers[-1], nn.ReLU)
    assert n_params == 10934

    epsilons = [
        linear_schedule(step, start_e=1.0, end_e=0.05, exploration_fraction=0.5, total_timesteps=500)
        for step in range(500)
    ]
    line(
        epsilons,
        labels={"x": "steps", "y": "epsilon"},
        title="Probability of random action",
        height=400,
        width=600,
    )
    tests.test_linear_schedule(linear_schedule)
    tests.test_epsilon_greedy_policy(epsilon_greedy_policy)

    env = Probe1(num_envs=4)
    assert env.single_observation_space.shape == (1,)
    assert env.single_action_space.shape == ()
    obs, _ = env.reset()
    assert obs.shape == (4, 1)

    tests.test_agent(DQNAgent)


def test_probe(probe_idx: int):
    """
    Tests a probe environment by training a network on it & verifying that the value functions are
    in the expected range.
    """
    # Train our network on this probe env.
    args = DQNArgs(
        env_id=f"Probe{probe_idx}-v0",
        wandb_project_name=f"test-probe-{probe_idx}",
        total_timesteps=3000 if probe_idx <= 2 else 5000,
        learning_rate=0.001,
        buffer_size=500,
        use_wandb=False,
        trains_per_target_update=20,
        video_log_freq=None,
        num_envs=1,
        steps_per_update=10,
        exploration_fraction=0.2,
        end_e=0.1,
    )
    trainer = DQNTrainer(args)
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
    print("Probe tests passed!\n")


if __name__ == "__main__":
    run_tests()
    for probe_idx in range(1, 6):
        test_probe(probe_idx)
