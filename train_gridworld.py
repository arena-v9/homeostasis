"""
Trains the DQN agent in the gridworld.

Every simple setting of GridworldConfig (envs/gridworld/config.py) is a flag with the same name (see flags.py), so
an experiment changes only what it passes and everything else keeps its default. For example:

    python train_gridworld.py --npc-enabled false --seed 1   # the agent alone, without the NPC
    python train_gridworld.py --npc-state-visible false --drop-npc-harm true --seed 2
    python train_gridworld.py --parallel-grid true --seed 3   # the NPC on a grid of its own

Every `--eval-every` training steps it evaluates the greedy policy on the same 64 fixed layouts (one episode each) and
records survival rate, deaths by cause, return, steps on fire, food eaten, pushes of the NPC, the NPC's steps on fire and
the NPC's survival rate, plus catastrophic-forgetting diagnostics (runs/diagnostics.py). It also records 16 full
episodes before training and at 25/50/75/100% of training. At the end it compares against a random policy and saves
the run as JSON in its own folder, ui/outputs/runs/<date-time>/ (see runs/outputs.py):
    run.json      settings (including the full config and the settings changed from the defaults) and final results
    curves.json   the evaluations and diagnostics over training
    replay.json   the recorded episodes
    q_network.pt  the trained Q-network's weights (for the evaluation harness)

To look at it: cd ui && python -m http.server 8000, then open http://localhost:8000/ and pick the run.
"""

import argparse
import time
from datetime import datetime

import torch as t
from tqdm.auto import tqdm

from dqn.args import DQNArgs
from dqn.trainer import DQNTrainer
from envs.gridworld.batched_env import BatchedGridworldEnv
from envs.gridworld.config import GridworldConfig
from flags import add_dataclass_flags, dataclass_from_flags
from runs import diagnostics
from runs.evaluation import evaluate, metric_labels
from runs.outputs import new_run_dir, register_run, write_json
from runs.recording import record_episodes, replay_data

NUM_ENVS = 16  # gridworlds run side by side during training (one gradient step per NUM_ENVS new transitions)
REPLAY_SEEDS = range(1000, 1016)  # the same 16 layouts at every checkpoint
REPLAY_CHECKPOINTS = (0.25, 0.5, 0.75, 1.0)  # fractions of training after which episodes are recorded


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=500, help="training steps between evaluations")
    add_dataclass_flags(parser, GridworldConfig, "gridworld config (defaults in envs/gridworld/config.py)")
    return parser.parse_args()


def make_dqn_args(cli: argparse.Namespace) -> DQNArgs:
    return DQNArgs(
        seed=cli.seed,
        total_timesteps=cli.timesteps,
        buffer_size=100_000,
        buffer_death_share=0.05,  # about 6 deaths in every batch, so the agent doesn't forget them (see ReplayBuffer)
        learning_starts=20_000,
        batch_size=128,
        learning_rate=5e-4,
        gamma=0.99,
        trains_per_target_update=200,
        exploration_fraction=0.3,
        start_e=1.0,
        end_e=0.05,
    )


def describe_changes(changes: dict) -> str:
    """A short label for a run's config, e.g. "npc_enabled=false" or "defaults"."""
    if not changes:
        return "defaults"
    format_value = lambda value: str(value).lower() if isinstance(value, bool) else value
    return ", ".join(f"{name}={format_value(value)}" for name, value in changes.items())


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
    # Catastrophic-forgetting diagnostics (runs/diagnostics.py), with their own random generator so they don't change
    # training
    probes = diagnostics.build_probes(config, device)
    diagnostics_rng = t.Generator(device=device).manual_seed(diagnostics.PROBE_SEED)
    evaluate_now = lambda: {
        "env_steps": trainer.agent.step,
        **evaluate(greedy, config, device),
        **diagnostics.diagnose(trainer, probes, diagnostics_rng),
    }
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
    if trainer.total_training_steps % eval_every != 0:
        history.append(evaluate_now())  # so the final results are those of the network that gets saved
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
    curves = {"metrics": metric_labels(config), "diagnostics": diagnostics.labels(config), "random": random_baseline,
              "history": history}
    write_json(run_dir / "curves.json", curves)
    write_json(run_dir / "replay.json", replay_data(replays))
    t.save(trainer.q_network.state_dict(), run_dir / "q_network.pt")
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
