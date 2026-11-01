"""
Trains the DQN agent in the gridworld.

Every simple setting of GridworldConfig (envs/gridworld/config.py) is a flag with the same name (see flags.py), so
an experiment changes only what it passes and everything else keeps its default. For example:

    python train_gridworld.py --npc-enabled false --seed 1   # the agent alone, without the NPC
    python train_gridworld.py --npc-state-visible false --witnessed true --seed 2
    python train_gridworld.py --npc-drive-weight 0.5

Every `--eval-every` training steps it evaluates the greedy policy on the same 64 fixed layouts (one episode each) and
records survival rate, return, steps on fire and food eaten. It also records 16 full episodes before training and at
25/50/75/100% of training. At the end it compares against a random policy and saves the run as JSON in its own
folder, ui/outputs/runs/<date-time>/ (see runs/outputs.py):
    run.json      settings (including the full config and the settings changed from the defaults) and final results
    curves.json   the evaluations over training
    replay.json   the recorded episodes

To look at it: cd ui && python -m http.server 8000, then open http://localhost:8000/ and pick the run.
"""

import argparse
import time
from datetime import datetime
from typing import Callable

import torch as t
from tqdm.auto import tqdm

from dqn.args import DQNArgs
from dqn.trainer import DQNTrainer
from envs.gridworld.batched_env import BatchedGridworldEnv
from envs.gridworld.config import GridworldConfig
from envs.gridworld.observation import CHANNEL_INDEX
from flags import add_dataclass_flags, dataclass_from_flags
from runs.outputs import new_run_dir, register_run, write_json
from runs.recording import record_episodes, replay_data

NUM_ENVS = 16  # gridworlds run side by side during training (one gradient step per NUM_ENVS new transitions)
EVAL_ENVS, EVAL_SEED = 64, 12345
REPLAY_SEEDS = range(1000, 1016)  # the same 16 layouts at every checkpoint
REPLAY_CHECKPOINTS = (0.25, 0.5, 0.75, 1.0)  # fractions of training after which episodes are recorded


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=500, help="training steps between evaluations")
    parser.add_argument("--wandb", action="store_true")
    add_dataclass_flags(parser, GridworldConfig, "gridworld config (defaults in envs/gridworld/config.py)")
    return parser.parse_args()


def make_dqn_args(cli: argparse.Namespace) -> DQNArgs:
    return DQNArgs(
        seed=cli.seed,
        total_timesteps=cli.timesteps,
        buffer_size=20_000,
        batch_size=128,
        learning_rate=5e-4,
        gamma=0.99,
        trains_per_target_update=200,
        exploration_fraction=0.3,
        start_e=1.0,
        end_e=0.05,
        use_wandb=cli.wandb,
        wandb_project_name="Gridworld",
    )


def describe_changes(changes: dict) -> str:
    """A short label for a run's config, e.g. "npc_enabled=false" or "defaults"."""
    if not changes:
        return "defaults"
    format_value = lambda value: str(value).lower() if isinstance(value, bool) else value
    return ", ".join(f"{name}={format_value(value)}" for name, value in changes.items())


def metric_labels(config: GridworldConfig) -> dict[str, str]:
    """The evaluation results, in the order they're reported, with their labels for the viewer's charts."""
    return {
        "survival": f"Survived all {config.max_steps} steps (fraction)",
        "return": "Episode return",
        "fire_steps": "Steps on fire per episode",
        "food_eaten": "Food eaten per episode",
    }


@t.inference_mode()
def evaluate(
    act_fn: Callable[[t.Tensor], t.Tensor],
    config: GridworldConfig,
    device: t.device,
) -> dict[str, float]:
    """
    Runs one full episode on each of EVAL_ENVS fixed layouts and averages over them. Returns the metrics named in
    `metric_labels`.
    """
    agent_energy = lambda obs: (obs[:, CHANNEL_INDEX["energy"]] * obs[:, CHANNEL_INDEX["self"]]).sum((-1, -2))
    env = BatchedGridworldEnv(EVAL_ENVS, EVAL_SEED, device, config)
    obs, _ = env.reset()
    alive = t.ones(EVAL_ENVS, dtype=t.bool, device=obs.device)
    episode_return, fire_steps, food_eaten = (t.zeros(EVAL_ENVS, device=obs.device) for _ in range(3))
    for _ in range(config.max_steps):
        next_obs, reward, terminated, truncated, infos = env.step(act_fn(obs))
        episode_return += reward * alive
        fire_steps += infos["agent_on_hazard"].float() * alive
        ate = agent_energy(next_obs) > agent_energy(obs)  # energy only goes up by eating
        food_eaten += (ate & ~terminated & alive).float()
        alive &= ~terminated  # count each env's first episode only
        obs = next_obs
    return {
        "survival": alive.float().mean().item(),
        "return": episode_return.mean().item(),
        "fire_steps": fire_steps.mean().item(),
        "food_eaten": food_eaten.mean().item(),
    }


def train_with_checkpoints(
    trainer: DQNTrainer,
    config: GridworldConfig,
    eval_every: int,
) -> tuple[list[dict], dict[str, list[dict]]]:
    """
    Fills the replay buffer, then trains, evaluating the greedy policy every `eval_every` training steps and recording
    episodes before training and at each of REPLAY_CHECKPOINTS. Returns the evaluation history (one dict per
    evaluation, with the env steps taken so far) and the recorded episodes by checkpoint label.
    """
    device = trainer.args.device
    greedy = lambda obs: trainer.q_network(obs).argmax(-1)
    evaluate_now = lambda: {"env_steps": trainer.agent.step, **evaluate(greedy, config, device)}
    record = lambda: record_episodes(greedy, config, REPLAY_SEEDS, device)
    replay_at = {int(trainer.total_training_steps * fraction): fraction for fraction in REPLAY_CHECKPOINTS}

    replays = {"untrained": record()}
    trainer.prepopulate_replay_buffer()
    history = [evaluate_now()]
    progress_bar = tqdm(range(trainer.total_training_steps))
    for step in progress_bar:
        trainer.add_to_replay_buffer(trainer.args.steps_per_update)
        trainer.training_step(step)
        steps_done = step + 1
        if steps_done % eval_every == 0:
            history.append(evaluate_now())
            progress_bar.set_postfix({key: f"{value:.2f}" for key, value in history[-1].items() if key != "env_steps"})
        if steps_done in replay_at:
            label = f"{replay_at[steps_done]:.0%} trained ({trainer.agent.step // 1000}k env steps)"
            replays[label] = record()
    return history, replays


def save_run(
    cli: argparse.Namespace,
    args: DQNArgs,
    config: GridworldConfig,
    trainer: DQNTrainer,
    history: list[dict],
    replays: dict[str, list[dict]],
    random_baseline: dict[str, float],
    train_seconds: float,
) -> None:
    """Saves the run to its own folder under ui/outputs/runs/ (see runs/outputs.py)."""
    changes = config.changes_from_defaults()
    final = history[-1]
    run_dir = new_run_dir()
    curves = {"metrics": metric_labels(config), "random": random_baseline, "history": history}
    write_json(run_dir / "curves.json", curves)
    write_json(run_dir / "replay.json", replay_data(replays))
    register_run(run_dir, {
        "created": datetime.now().isoformat(timespec="seconds"),
        "title": f"{describe_changes(changes)}; seed {cli.seed}; {trainer.agent.step // 1000}k env steps",
        "summary": (f"Trained for {trainer.agent.step:,} environment steps with config: {describe_changes(changes)}. "
                    f"The NPC is {'on the map' if config.npc_enabled else 'not on the map'}. The same "
                    f"{len(REPLAY_SEEDS)} layouts are replayed at each training stage. The reward is the reduction in "
                    "the agent's drive (its distance from full health and energy); see DESIGN.md."),
        "script": "train_gridworld.py",
        "train_seconds": round(train_seconds),
        "final": final,
        "random": random_baseline,
        "num_envs": NUM_ENVS,
        "config_changes": changes,
        "config": config.to_dict(),
        "args": {key: str(value) if key == "device" else value for key, value in vars(args).items()},
    })
    print(f"saved {run_dir}; view all runs with: cd ui && python -m http.server 8000, then open http://localhost:8000/")


def main():
    cli = parse_cli()
    config = dataclass_from_flags(cli, GridworldConfig)
    print("config changes:", describe_changes(config.changes_from_defaults()))
    args = make_dqn_args(cli)
    trainer = DQNTrainer(args, BatchedGridworldEnv(NUM_ENVS, args.seed, args.device, config))

    num_actions = trainer.envs.single_action_space.n
    random_policy = lambda obs: t.randint(0, num_actions, (obs.shape[0],), device=obs.device)
    random_baseline = evaluate(random_policy, config, args.device)
    print("random policy:", {key: round(value, 2) for key, value in random_baseline.items()})

    start_time = time.time()
    history, replays = train_with_checkpoints(trainer, config, cli.eval_every)
    train_seconds = time.time() - start_time
    print(f"trained {trainer.agent.step} env steps in {train_seconds:.0f}s")

    print(f"{'':12}{'random':>10}{'trained':>10}")
    for key in metric_labels(config):
        print(f"{key:12}{random_baseline[key]:>10.2f}{history[-1][key]:>10.2f}")

    save_run(cli, args, config, trainer, history, replays, random_baseline, train_seconds)


if __name__ == "__main__":
    main()
