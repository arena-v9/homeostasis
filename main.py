"""
Runs the tests and training runs from the 2.2.1 DQN notebook. The action-gap diagnostic re-trains CartPole
from scratch with extra logging, so it's guarded by `SLOW`.
"""

import torch as t
import torch.nn as nn
from tqdm.auto import tqdm

import tests
from cartpole import CartPole
from common import device
from dqn_agent import DQNAgent
from dqn_args import DQNArgs
from dqn_trainer import DQNTrainer
from exploration import epsilon_greedy_policy, linear_schedule
from plotting import line, plot_cartpole_obs_and_dones
from probe_envs import Probe1
from q_network import QNetwork
from replay_buffer import ReplayBuffer

# Flip this to True to run the action-gap diagnostic.
SLOW = False


def run_tests():
    envs = CartPole(num_envs=4, device=device)
    print(envs.single_action_space)  # 2 actions: left and right
    print(envs.single_observation_space)  # Box(4): each observation can take a continuous range of values
    obs, infos = envs.reset()
    print(obs.shape, obs.device)  # (num_envs, 4): one observation per environment, as a tensor on our device

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


def demo_replay_buffer():
    buffer = ReplayBuffer(num_envs=1, obs_shape=(4,), action_shape=(), buffer_size=256, seed=0, device=device)
    envs = CartPole(num_envs=1, seed=0, device=device)
    obs, infos = envs.reset()

    for i in range(256):
        # Choose random actions (one per environment), and take a step in the environment
        actions = t.randint(0, envs.single_action_space.n, (envs.num_envs,), device=device)
        next_obs, rewards, terminated, truncated, infos = envs.step(actions)

        # Get `true_next_obs` by finding all environments where we terminated & replacing `next_obs`
        # with the actual terminal states (the env has already auto-reset those environments, and stored
        # the observation they ended on in `infos["final_observation"]`)
        done = terminated | truncated
        true_next_obs = next_obs.clone()
        true_next_obs[done] = infos["final_observation"][done]

        # Add experience to buffer
        buffer.add(obs, actions, rewards, terminated, true_next_obs)
        obs = next_obs

    sample = buffer.sample(256)

    plot_cartpole_obs_and_dones(
        buffer.obs.cpu(),
        buffer.terminated.cpu(),
        title="Current obs s<sub>t</sub><br>so when d<sub>t+1</sub> = 1, these are the states just before termination",
    )
    plot_cartpole_obs_and_dones(
        buffer.next_obs.cpu(),
        buffer.terminated.cpu(),
        title="Next obs s<sub>t+1</sub><br>so when d<sub>t+1</sub> = 1, these are the terminated states",
    )
    plot_cartpole_obs_and_dones(
        sample.obs.cpu(),
        sample.terminated.cpu(),
        title="Current obs s<sub>t</sub> (sampled)<br>this is what gets fed into our model for training",
    )


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


def train_cartpole():
    args = DQNArgs(use_wandb=False)  # the notebook uses use_wandb=True; flip it if you're logged in to wandb
    trainer = DQNTrainer(args)
    trainer.train()


def action_gap_demo():
    """
    Re-run training with the default settings, but every 500 gradient steps record:
      1. how long the greedy policy survives on 16 fresh environments (no exploration)
      2. how many of the transitions in the replay buffer are terminal, i.e. does the data still contain failures
      3. the action gap  max_a Q(s,a) - min_a Q(s,a)  averaged over a fixed batch of states from the random-policy phase
      4. max_a Q(s, a) on a fixed set of doomed states, from which every action terminates next step (correct value 0)
      5. the TD loss, recomputed on a large sample from the buffer (the same quantity training_step minimises)
    """
    from plotly.subplots import make_subplots

    from video import greedy_rollout

    args = DQNArgs(use_wandb=False, steps_per_live_video=None)
    trainer = DQNTrainer(args)
    trainer.prepopulate_replay_buffer()
    probe_obs = trainer.buffer.sample(2048).obs  # sampled now, while the buffer is still full of random-policy failures
    pre_terminal_obs = trainer.buffer.obs[trainer.buffer.terminated]  # states whose taken action ended the episode
    # Check these are doomed under BOTH actions, by writing each state into a batched env and stepping it with each action.
    check_env = CartPole(num_envs=len(pre_terminal_obs), device=args.device)
    doomed = t.ones(len(pre_terminal_obs), dtype=t.bool, device=args.device)
    for a in (0, 1):
        check_env.state = pre_terminal_obs.clone()
        doomed &= check_env.step(t.full((len(pre_terminal_obs),), a, device=args.device))[2]  # terminated flag
    print(f"{doomed.sum().item()} of {len(doomed)} pre-termination states are doomed under both actions")
    doomed_obs = pre_terminal_obs[doomed]
    greedy = lambda obs: trainer.q_network(obs).argmax(-1)

    def td_loss(n_samples: int = 2048) -> float:
        d = trainer.buffer.sample(n_samples)
        with t.inference_mode():
            target = d.rewards + args.gamma * trainer.target_network(d.next_obs).max(-1).values * (1 - d.terminated.float())
            predicted = trainer.q_network(d.obs).gather(-1, d.actions.unsqueeze(-1)).squeeze(-1)
        return (target - predicted).pow(2).mean().item()

    log = []
    pbar = tqdm(range(args.total_training_steps))
    for step in pbar:
        trainer.add_to_replay_buffer(args.steps_per_update)
        trainer.training_step(step)
        if (step + 1) % 500 == 0:
            _, _, lifespan = greedy_rollout(CartPole(num_envs=16, seed=step, device=args.device), greedy)
            with t.inference_mode():
                q = trainer.q_network(probe_obs)
                q_doomed = trainer.q_network(doomed_obs).max(-1).values
            record = {
                "G": lifespan.mean().item(),
                "n_term": trainer.buffer.terminated.sum().item(),
                "act_gap": (q.max(-1).values - q.min(-1).values).mean().item(),
                "q_doom": q_doomed.mean().item(),
                "loss": td_loss(),
            }
            log.append([step + 1, *record.values()])
            pbar.set_postfix({k: f"{v:.3g}" for k, v in record.items()})

    steps, greedy_lifespan, n_terminal, action_gap, q_doomed, td_loss_curve = zip(*log)
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.06, subplot_titles=[
        "Greedy episode length on 16 fresh envs",
        "Terminal transitions in the replay buffer (log scale; 0 shown at the bottom)",
        "Action gap  max_a Q(s,a) - min_a Q(s,a)  on a fixed batch of states (log scale)",
        "max_a Q(s, a) on fixed doomed states  (correct value 0)",
        "TD loss on 2048 sampled transitions (log scale)",
    ])
    n_terminal_plot = [max(n, 0.5) for n in n_terminal]  # a log axis can't show 0, so exact zeros sit at 0.5 with a tick labelled "0"
    for row, y in enumerate([greedy_lifespan, n_terminal_plot, action_gap, q_doomed, td_loss_curve], start=1):
        fig.add_scatter(x=steps, y=y, mode="lines+markers", showlegend=False, row=row, col=1)
    ticks = [0.5, 1, 10, 100, 1000]
    fig.update_yaxes(type="log", tickvals=ticks, ticktext=["0"] + [str(v) for v in ticks[1:]], range=[-0.4, 3], row=2, col=1)
    fig.update_yaxes(type="log", row=3, col=1)
    fig.update_yaxes(type="log", row=5, col=1)
    fig.update_layout(height=1100, width=800, xaxis5_title="gradient steps", margin=dict(t=40, b=40, l=40, r=20))
    fig.show()


if __name__ == "__main__":
    run_tests()
    demo_replay_buffer()
    for probe_idx in range(1, 6):
        test_probe(probe_idx)
    train_cartpole()
    if SLOW:
        action_gap_demo()
