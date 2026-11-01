"""
Trains the DQN agent alone in the gridworld (no NPC), to check the environment works and the agent learns to
survive: eat to keep its energy up and stay off fire.

Every `--eval-every` training steps it evaluates the greedy policy on the same 64 fixed layouts (one 100-step
episode each) and records survival rate, return, steps on fire and food eaten. It also records 16 full episodes
before training and at 25/50/75/100% of training. At the end it compares against a random policy and writes to
`outputs/`:
    gridworld_solo_replay.html   step-through replays of the recorded episodes (open in a browser)
    gridworld_solo_curves.html   learning curves

Usage: python train_gridworld_solo.py [--timesteps 200000] [--seed 1] [--wandb]
"""

import argparse
import time
from pathlib import Path

import torch as t
from plotly.subplots import make_subplots
from tqdm.auto import tqdm

from dqn_args import DQNArgs
from dqn_trainer import DQNTrainer
from env_registry import make_envs
from gridworld import CHANNELS
from gridworld_config import GridworldConfig
from replay_viewer import record_episodes, write_replay_html

ENV_ID = "Gridworld-solo-v0"
CONFIG = GridworldConfig(npc_enabled=False)  # must match the registered env
OUT_DIR = Path(__file__).parent / "outputs"
EVAL_ENVS, EVAL_SEED, EPISODE_STEPS = 64, 12345, 100
REPLAY_SEEDS = range(1000, 1016)  # the same 16 layouts at every checkpoint
ENERGY, SELF = CHANNELS.index("energy"), CHANNELS.index("self")


@t.inference_mode()
def evaluate(act_fn, device) -> dict[str, float]:
    """Runs one 100-step episode on each of EVAL_ENVS fixed layouts and averages over them."""
    env = make_envs(ENV_ID, EVAL_ENVS, EVAL_SEED, device)
    obs, _ = env.reset()
    alive = t.ones(EVAL_ENVS, dtype=t.bool, device=obs.device)
    episode_return, fire_steps, food_eaten = (t.zeros(EVAL_ENVS, device=obs.device) for _ in range(3))
    for _ in range(EPISODE_STEPS):
        energy = (obs[:, ENERGY] * obs[:, SELF]).sum((-1, -2))
        next_obs, reward, terminated, truncated, infos = env.step(act_fn(obs))
        next_energy = (next_obs[:, ENERGY] * next_obs[:, SELF]).sum((-1, -2))
        episode_return += reward * alive
        fire_steps += infos["agent_on_hazard"].float() * alive
        food_eaten += ((next_energy > energy) & ~terminated & alive).float()  # energy only goes up by eating
        alive &= ~terminated  # count each env's first episode only
        obs = next_obs
    return {
        "survival": alive.float().mean().item(),
        "return": episode_return.mean().item(),
        "fire_steps": fire_steps.mean().item(),
        "food_eaten": food_eaten.mean().item(),
    }


def save_curves(history: list[dict], random_baseline: dict, path: Path):
    metrics = [
        ("survival", "Survived all 100 steps (fraction)"),
        ("return", "Episode return"),
        ("fire_steps", "Steps on fire per episode"),
        ("food_eaten", "Food eaten per episode"),
    ]
    fig = make_subplots(rows=len(metrics), cols=1, shared_xaxes=True, subplot_titles=[m[1] for m in metrics])
    steps = [h["env_steps"] for h in history]
    for row, (key, _) in enumerate(metrics, start=1):
        fig.add_scatter(x=steps, y=[h[key] for h in history], mode="lines+markers", name="greedy DQN",
                        line_color="#1f77b4", showlegend=row == 1, row=row, col=1)
        fig.add_hline(y=random_baseline[key], line_dash="dash", line_color="grey", row=row, col=1,
                      annotation_text="random policy" if row == 1 else None)
    fig.update_layout(height=1000, width=800, title="DQN agent alone in the gridworld", xaxis4_title="environment steps")
    fig.write_html(path, include_plotlyjs="cdn")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=500, help="training steps between evaluations")
    parser.add_argument("--wandb", action="store_true")
    cli = parser.parse_args()

    args = DQNArgs(
        env_id=ENV_ID,
        seed=cli.seed,
        num_envs=16,
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
        wandb_project_name="GridworldSolo",
        steps_per_live_video=None,
    )
    OUT_DIR.mkdir(exist_ok=True)
    trainer = DQNTrainer(args)
    greedy = lambda obs: trainer.q_network(obs).argmax(-1)

    random_baseline = evaluate(lambda obs: t.randint(0, 5, (obs.shape[0],), device=obs.device), args.device)
    print("random policy:", {k: round(v, 2) for k, v in random_baseline.items()})

    record = lambda: record_episodes(greedy, CONFIG, REPLAY_SEEDS, args.device)
    replays = {"untrained": record()}
    replay_at = {args.total_training_steps * k // 4: f"{k * 25}% trained" for k in (1, 2, 3, 4)}

    t0 = time.time()
    trainer.prepopulate_replay_buffer()
    history = [{"env_steps": trainer.agent.step, **evaluate(greedy, args.device)}]
    pbar = tqdm(range(args.total_training_steps))
    for step in pbar:
        trainer.add_to_replay_buffer(args.steps_per_update)
        trainer.training_step(step)
        if (step + 1) % cli.eval_every == 0:
            history.append({"env_steps": trainer.agent.step, **evaluate(greedy, args.device)})
            pbar.set_postfix({k: f"{v:.2f}" for k, v in history[-1].items() if k != "env_steps"})
        if step + 1 in replay_at:
            replays[f"{replay_at[step + 1]} ({trainer.agent.step // 1000}k env steps)"] = record()
    print(f"trained {trainer.agent.step} env steps in {time.time() - t0:.0f}s")

    final = history[-1]
    print(f"{'':12}{'random':>10}{'trained':>10}")
    for key in ("survival", "return", "fire_steps", "food_eaten"):
        print(f"{key:12}{random_baseline[key]:>10.2f}{final[key]:>10.2f}")

    save_curves(history, random_baseline, OUT_DIR / "gridworld_solo_curves.html")
    write_replay_html(
        replays,
        OUT_DIR / "gridworld_solo_replay.html",
        title="Solo Gridworld Replays",
        subtitle=(f"The DQN agent alone in the gridworld (no NPC), trained for {trainer.agent.step:,} environment steps. "
                  "The same 16 layouts are replayed at each training stage. The reward is the reduction in the agent's "
                  "drive (its distance from full health and energy); see DESIGN.md."),
    )
    print(f"wrote {OUT_DIR}/gridworld_solo_replay.html and gridworld_solo_curves.html")


if __name__ == "__main__":
    main()
