"""
Runs the tests and training runs from the 2.3 PPO notebook. The bonus training loops (Atari / MuJoCo /
MountainCar / Pendulum / swing-up) each take a few minutes, so they're guarded by `SLOW`.
"""

import numpy as np
import torch as t
from IPython.display import display

from common import ENV_DICT, CartPole, MountainCar, device  # common must be imported first (sets up sys.path)
import part3_ppo.tests as tests
from easy_cart import EasyCart
from my_pendulum import MyPendulum
from networks import get_actor_and_critic
from objectives import calc_clipped_surrogate_objective, calc_entropy_bonus, calc_value_function_loss
from part3_ppo.utils import arg_help
from plotly_utils import plot_cartpole_obs_and_dones
from ppo_agent import PPOAgent
from ppo_args import PPOArgs
from ppo_scheduler import PPOScheduler
from ppo_trainer import PPOTrainer
from ppo_trainer_cts import PPOTrainerCts
from replay_memory import ReplayMemory, compute_advantages, get_minibatch_indices
from rl_utils import record_brax_video, record_grid_video
from spin_cart import SpinCart

# Flip this to True to actually run the bonus environments.
SLOW = False

# Register the reward-equipped subclass so `PPOTrainer`/`PPOTrainerCts` build it for mode="pendulum"
# (the base `Pendulum` deliberately raises until you supply a reward).
ENV_DICT["pendulum"] = MyPendulum


def run_tests():
    args = PPOArgs(num_minibatches=2)  # changing this also changes minibatch_size and total_training_steps
    arg_help(args)

    tests.test_get_actor_and_critic(get_actor_and_critic, mode="classic-control")
    tests.test_compute_advantages(compute_advantages)

    rng = np.random.default_rng(0)
    batch_size = 12
    minibatch_size = 6
    # num_minibatches = batch_size // minibatch_size = 2
    indices = get_minibatch_indices(rng, batch_size, minibatch_size)
    assert isinstance(indices, list)
    assert all(isinstance(x, np.ndarray) for x in indices)
    assert np.array(indices).shape == (2, 6)
    assert sorted(np.unique(indices)) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    print("All tests in `test_minibatch_indexes` passed!")

    tests.test_ppo_agent(PPOAgent)
    tests.test_calc_clipped_surrogate_objective(calc_clipped_surrogate_objective)
    tests.test_calc_value_function_loss(calc_value_function_loss)
    tests.test_calc_entropy_bonus(calc_entropy_bonus)
    tests.test_ppo_scheduler(PPOScheduler)
    tests.test_get_actor_and_critic(get_actor_and_critic, mode="atari")
    tests.test_get_actor_and_critic(get_actor_and_critic, mode="mujoco")


def demo_replay_memory():
    num_steps_per_rollout = 128
    num_envs = 2
    batch_size = num_steps_per_rollout * num_envs  # 256

    minibatch_size = 128
    num_minibatches = batch_size // minibatch_size  # 2

    batches_per_learning_phase = 2

    envs = CartPole(num_envs=num_envs)  # GPU-batched env: reset()/step() return on-device tensors
    memory = ReplayMemory(num_envs, (4,), (), batch_size, minibatch_size, batches_per_learning_phase)

    logprobs = values = t.zeros(num_envs, device=device)  # dummy values, just so we can see demo of plot
    obs, _ = envs.reset()
    terminated = t.zeros(num_envs, dtype=t.bool, device=device)  # d_t for the very first obs

    for i in range(num_steps_per_rollout):
        # Choose a random action and step the env (actions are an on-device tensor for the batched env)
        actions = t.randint(0, envs.single_action_space.n, (num_envs,), device=device)
        next_obs, rewards, next_terminated, truncated, infos = envs.step(actions)

        # Add experience to memory (everything is already an on-device tensor). Like `PPOAgent.play_step`,
        # the flag we store next to s_t is d_t, i.e. "s_t is the first obs of a new episode".
        memory.add(obs, actions, logprobs, values, rewards, terminated)
        obs, terminated = next_obs, next_terminated

    plot_cartpole_obs_and_dones(
        t.stack(memory.obs).cpu(),
        t.stack(memory.terminated).cpu(),
        title="Current obs s<sub>t</sub><br>Dotted lines indicate d<sub>t</sub> = 1, solid lines are environment separators",
    )

    next_value = next_done = t.zeros(envs.num_envs).to(device)  # dummy values, just so we can see demo of plot
    minibatches = memory.get_minibatches(next_value, next_done, gamma=0.99, gae_lambda=0.95)

    plot_cartpole_obs_and_dones(
        minibatches[0].obs.cpu(),
        minibatches[0].terminated.cpu(),
        title="Current obs (sampled)<br>this is what gets fed into our model for training",
    )


def test_probe(probe_idx: int):
    """
    Tests a probe environment by training a network on it & verifying that the value functions are
    in the expected range.
    """
    # Train our network
    args = PPOArgs(
        env_id=f"Probe{probe_idx}-v0",
        mode="probe",  # use the GPU (tensor-native) probe envs, so PPOTrainer's GPU agent can run them
        wandb_project_name=f"test-probe-{probe_idx}",
        total_phases=14,  # adjust up if needed to make tests pass
        num_envs=256,
        num_steps_per_rollout=8,
        num_minibatches=4,
        lr=0.01, # also maybe adjust this
        video_log_freq=None,
        use_wandb=False,
    )
    trainer = PPOTrainer(args)
    trainer.train()
    agent = trainer.agent

    # Get the correct set of observations, and corresponding values we expect
    obs_for_probes = [[[0.0]], [[-1.0], [+1.0]], [[0.0], [1.0]], [[0.0]], [[0.0], [1.0]]]
    expected_value_for_probes = [
        [[1.0]],
        [[-1.0], [+1.0]],
        [[args.gamma], [1.0]],
        [[1.0]],
        [[1.0], [1.0]],
    ]
    expected_probs_for_probes = [None, None, None, [[0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]
    tolerances = [1e-3, 1e-3, 1e-3, 2e-3, 2e-3]
    obs = t.tensor(obs_for_probes[probe_idx - 1]).to(args.device)

    # Calculate the actual value & probs, and verify them
    with t.inference_mode():
        value = agent.critic(obs)
        probs = agent.actor(obs).softmax(-1)
    # A bf16 network (e.g. under autocast) can't represent these targets to better than its eps
    atol = max(tolerances[probe_idx - 1], t.finfo(value.dtype).eps)
    expected_value = t.tensor(expected_value_for_probes[probe_idx - 1]).to(args.device)
    t.testing.assert_close(value.float(), expected_value, atol=atol, rtol=0)
    expected_probs = expected_probs_for_probes[probe_idx - 1]
    if expected_probs is not None:
        t.testing.assert_close(probs.float(), t.tensor(expected_probs).to(args.device), atol=atol, rtol=0)
    print("Probe tests passed!\n")


def train_cartpole():
    args = PPOArgs(use_wandb=False, video_log_freq=50)
    trainer = PPOTrainer(args)
    trainer.train()
    display(record_grid_video(trainer, kind="classic-control"))


def train_easy_cart():
    # Swap the shaped env into ENV_DICT so `mode="classic-control"` builds it (we restore the
    # unshaped CartPole afterwards). `env_id` is just a label for the run name.
    ENV_DICT["classic-control"] = EasyCart
    args = PPOArgs(env_id="EasyCart", use_wandb=False, video_log_freq=50)
    trainer = PPOTrainer(args)
    trainer.train()
    ENV_DICT["classic-control"] = CartPole
    display(record_grid_video(trainer, kind="classic-control"))


def train_spin_cart():
    ENV_DICT["classic-control"] = SpinCart
    # The shaped spinning reward is less stable than plain CartPole once the policy has spun up, so the
    # final video occasionally shows a policy that spun up and then partly lost it. It still spins, and
    # the run takes about 10 s on a CPU.
    args = PPOArgs(env_id="SpinCart", use_wandb=False, video_log_freq=50)
    trainer = PPOTrainer(args)
    trainer.train()
    ENV_DICT["classic-control"] = CartPole  # restore the unshaped env
    display(record_grid_video(trainer, kind="classic-control"))


def train_breakout():
    args = PPOArgs(
        env_id="ALE/Breakout-v5",
        wandb_project_name="PPOAtari",
        mode="atari",
        total_phases=183,  # 183 * 8192 = 1.5M env steps
        num_envs=256,
        num_steps_per_rollout=32,
        num_minibatches=4,
        lr=2.5e-4,
        clip_coef=0.1,
        ent_coef=0.01,
        vf_coef=0.5,
    )
    trainer = PPOTrainer(args)
    trainer.train()

    # A 4x4 grid video of 16 of the trained agent's Breakout games (cells flash red when an env resets):
    display(record_grid_video(trainer, kind="atari"))


def train_pong():
    args = PPOArgs(
        env_id="ALE/Pong-v5",
        wandb_project_name="PPOAtari",
        mode="atari",
        total_phases=366,  # 366 * 8192 = 3M env steps
        num_envs=256,
        num_steps_per_rollout=32,
        num_minibatches=4,
        lr=2.5e-4,
        clip_coef=0.1,
        ent_coef=0.01,
        vf_coef=0.5,
    )
    trainer = PPOTrainer(args)
    trainer.train()
    display(record_grid_video(trainer, kind="atari"))


def train_hopper():
    args = PPOArgs(
        env_id="Hopper-v4",
        wandb_project_name="PPOMuJoCo",
        mode="mujoco",
        total_phases=200,  # solves (return >= 1000) after ~60-120 phases; 64 envs is as fast as 2048 and runs on a CPU
        num_envs=64,
        num_steps_per_rollout=32,
        num_minibatches=8,
        lr=1e-3,
        gamma=0.97,
        ent_coef=0.0,
        vf_coef=0.5,
    )
    trainer = PPOTrainerCts(args)
    trainer.train()
    # gym-style MP4 of the trained agent (tracking camera; in-browser viewer fallback)
    display(record_brax_video(trainer.envs, trainer.agent.actor))


def train_humanoid():
    args = PPOArgs(
        env_id="Humanoid-v4",
        wandb_project_name="PPOMuJoCo",
        mode="mujoco",
        total_phases=228,
        num_envs=2048,
        num_steps_per_rollout=32,
        num_minibatches=8,
        lr=1e-3,
        ent_coef=0.0,
        vf_coef=0.5,
    )
    trainer = PPOTrainerCts(args)
    trainer.train()
    display(record_brax_video(trainer.envs, trainer.agent.actor))


def train_mountain_car():
    # Shape MountainCar's sparse -1 reward by rewarding total mechanical energy (kinetic + potential).
    # The agent can only raise this by pumping energy into the system, which is how it escapes the
    # valley. The dynamics are unchanged; we just scale to O(1) for a well-conditioned value function.
    def mountaincar_energy_reward(self, action):
        x, v = self.state[:, 0], self.state[:, 1]
        kinetic = 0.5 * v ** 2
        potential = (self.gravity / 3.0) * t.sin(3 * x)  # gravitational PE consistent with the dynamics
        return 1000.0 * (kinetic + potential)

    MountainCar.reward_function = mountaincar_energy_reward

    args = PPOArgs(
        env_id="MountainCar-v0",
        wandb_project_name="PPOMountainCar",
        mode="mountain-car",
        total_phases=585,  # reaches the flag after ~150-400 phases. This task wants updates, not samples: 2048 envs never got there
        num_envs=16,
        num_steps_per_rollout=64,
        num_minibatches=4,
        lr=3e-3,
        gamma=0.99,
        ent_coef=0.01,
        vf_coef=0.5,
    )
    trainer = PPOTrainer(args)
    trainer.train()
    display(record_grid_video(trainer, kind="mountain-car"))


def train_pendulum():
    args = PPOArgs(
        env_id="Pendulum-v1",
        wandb_project_name="PPOPendulum",
        mode="pendulum",
        total_phases=240,  # balances after ~100-160 phases (about 10 s on a CPU thread)
        num_envs=64,
        num_steps_per_rollout=64,
        num_minibatches=4,
        lr=3e-3,
        gamma=0.95,
        ent_coef=0.0,
        vf_coef=0.5,
    )
    trainer = PPOTrainerCts(args)
    trainer.train()
    display(record_grid_video(trainer, kind="pendulum"))


def train_swing_up():
    args = PPOArgs(
        env_id="CartDoublePendulum",
        wandb_project_name="PPOSwingUp",
        mode="swing-up",
        total_phases=228,
        num_envs=4096,
        num_steps_per_rollout=64,
        num_minibatches=8,
        batches_per_learning_phase=4,
        lr=5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        ent_coef=0.003,
        vf_coef=0.5,
    )
    trainer = PPOTrainerCts(args)
    trainer.train()
    # 4x4 grid of independent carts, each rendered from its own observation.
    display(record_grid_video(trainer, kind="swingup"))


if __name__ == "__main__":
    run_tests()
    demo_replay_memory()
    for probe_idx in range(1, 6):
        test_probe(probe_idx)
    train_cartpole()
    train_easy_cart()
    train_spin_cart()
    if SLOW:
        train_breakout()
        train_pong()
        train_hopper()
        train_humanoid()
        train_mountain_car()
        train_pendulum()
        train_swing_up()
