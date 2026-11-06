"""
Slide figures of experiments 1 and 2, from the analysis output (ui/outputs/experiments/experiment-{1,2}.json, written
by `python -m runs.experiments experiment-N --allow-incomplete`). Per-run values and p-values are the analysis's own;
only the 95% intervals of each group's mean are bootstrapped here. PNGs in presentation/out/.

    python presentation/make_result_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS, OUT = ROOT / "ui" / "outputs" / "experiments", Path(__file__).resolve().parent / "out"
# Data-viz reference palette (validated): experiment 1's groups take the first three categorical slots; experiment 2
# colours the agent model (homeostatic blue, psychopath red), filled when trained with the NPC, hollow without
GROUP_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
HOMEOSTATIC, PSYCHOPATH = "#2a78d6", "#e34948"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True, "legend.frameon": False,
    "lines.linewidth": 2,
})


def load(n: int) -> dict:
    return json.loads((EXPERIMENTS / f"experiment-{n}.json").read_text())


def per_run(question: dict, group: int, measure: str) -> np.ndarray:
    return np.array([run[measure] for run in question["per_run"][group]], dtype=float)


def p_value(question: dict, a: int, b: int, measure: str) -> float:
    cell = question["differences"][a][b] or question["differences"][b][a]
    return cell[measure]["p_value"]


def bootstrap_ci(values: np.ndarray, rng) -> tuple[float, float]:
    return tuple(np.percentile(rng.choice(values, (20_000, len(values))).mean(axis=1), [2.5, 97.5]))


def group_marks(ax, x, values, color, rng, filled=True, fmt="{:.0f}", scale=1.0):
    v = values * scale
    ax.scatter(x + np.linspace(-0.07, 0.07, len(v)), v, s=34, facecolor=color if filled else SURFACE, edgecolor=color,
               linewidths=1.5, alpha=0.7, zorder=3)
    low, high = bootstrap_ci(v, rng)
    ax.plot([x, x], [low, high], color=color, lw=2, zorder=4)
    ax.plot([x - 0.13, x + 0.13], [v.mean()] * 2, color=color, lw=3, zorder=5)
    ax.text(x + 0.17, v.mean(), fmt.format(v.mean()), color=INK2, fontsize=10.5, va="center")


def bracket(ax, x1, x2, y, text, clear):
    ax.plot([x1, x1, x2, x2], [y - 1.2, y, y, y - 1.2], color=INK2 if clear else MUTED, lw=1)
    ax.text((x1 + x2) / 2, y + 0.8, text, ha="center", va="bottom", fontsize=10, color=INK if clear else MUTED,
            fontweight="bold" if clear else "normal")


def title(fig, text, sub):
    fig.text(0.04, 0.95, text, fontsize=15, fontweight="bold", va="top")
    fig.text(0.04, 0.885, sub, fontsize=10.5, color=INK2, va="top")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)
    print("wrote", OUT / name)


def experiment_1_reliability():
    d = load(1)
    q = next(q for q in d["questions"] if q["kind"] == "better")
    rng = np.random.default_rng(0)
    labels = [g["label"] for g in d["groups"]]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    fig.subplots_adjust(left=0.1, right=0.95, top=0.76, bottom=0.12)
    for i, color in enumerate(GROUP_COLORS):
        group_marks(ax, i, per_run(q, i, "reliability"), color, rng, fmt="{:.0f}%", scale=100)
    for a, b, y in ((0, 1, 104), (1, 2, 112)):
        p = p_value(q, a, b, "reliability")
        bracket(ax, a, b, y, f"p = {p:.3f}", p < 0.05)
    ax.set_xticks(range(3), labels)
    ax.tick_params(axis="x", colors=INK2, labelsize=11, length=0)
    ax.grid(axis="x", visible=False)
    ax.set_xlim(-0.5, 2.6)
    ax.set_ylim(0, 120)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("Reliability: evaluations after 100k steps\nwith at least 95% survival")
    runs = len(q["per_run"][0])
    title(fig, "Experiment 1: watching an NPC makes the agent more reliable at surviving",
          f"The NPC lives on a grid the agent can never enter; its state never enters the reward. {runs} runs per group "
          "(dots);\nbars: mean with 95% bootstrap interval. Starvation deaths per 1,000 test episodes: "
          + ", ".join(f"{m:.0f}" for m in q["means"]["deaths_starvation"]) + ".")
    save(fig, "results_1_exp1_reliability.png")


def experiment_1_survival():
    d = load(1)
    fig, ax = plt.subplots(figsize=(10, 5.6))
    fig.subplots_adjust(left=0.1, right=0.8, top=0.8, bottom=0.13)
    ends = []
    for group, color in zip(d["groups"], GROUP_COLORS):
        steps = np.array(group["curves"]["env_steps"]) / 1000
        survival = 100 * np.array(group["curves"]["survival"]["mean"])
        ax.plot(steps, survival, color=color, zorder=3)
        ends.append((survival[-1], group["label"], color))
    # End labels spread so they don't overlap (at least 5 points apart), each with a leader to its line
    placed = []
    for y, label, color in sorted(ends, reverse=True):
        y_text = min(y, placed[-1] - 5) if placed else y
        placed.append(y_text)
        ax.annotate(label, xy=(steps[-1], y), xytext=(steps[-1] + 6, y_text), color=INK2, fontsize=10, va="center",
                    annotation_clip=False, arrowprops={"arrowstyle": "-", "color": color, "lw": 1})
    ax.axvline(100, color=AXIS, lw=1)
    ax.text(102, 6, "reliability is measured\nafter 100k steps", fontsize=9, color=MUTED)
    ax.set_xlim(steps[0], steps[-1])
    ax.set_ylim(0, 102)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Environment steps (k)")
    ax.set_ylabel("Survival (share of 64 test layouts)")
    title(fig, "All groups learn to survive; without the NPC they stay a little lower",
          "Mean survival of the greedy policy at each evaluation, across runs. Averaging hides individual collapses; "
          "the reliability\nmeasure (previous figure) counts them.")
    save(fig, "results_2_exp1_survival.png")


def experiment_2_pushes():
    d = load(2)
    q = d["questions"][0]
    rng = np.random.default_rng(1)
    styles = [(PSYCHOPATH, False), (PSYCHOPATH, True), (HOMEOSTATIC, True), (HOMEOSTATIC, False)]
    labels = ["Psychopath,\nno NPC", "Psychopath,\nNPC", "Homeostatic,\nNPC", "Homeostatic,\nno NPC"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.74, bottom=0.15, wspace=0.22)
    for ax, (measure, name, top) in zip(axes, (("pushes", "All pushes of the NPC per test episode", 3.2),
                                                ("harmful_pushes", "Pushes of the NPC onto fire per test episode", 0.34))):
        for i, (color, filled) in enumerate(styles):
            group_marks(ax, i, per_run(q, i, measure), color, rng, filled=filled, fmt="{:.2f}")
        p = p_value(q, 1, 2, measure)
        y = top * 0.9
        ax.plot([1, 1, 2, 2], [y - top * 0.03, y, y, y - top * 0.03], color=INK2 if p < 0.05 else MUTED, lw=1)
        ax.text(1.5, y + top * 0.02, f"p = {p:.3f}", ha="center", va="bottom", fontsize=10,
                color=INK if p < 0.05 else MUTED, fontweight="bold" if p < 0.05 else "normal")
        ax.set_title(name, fontsize=11.5, loc="left", color=INK)
        ax.set_xticks(range(4), labels)
        ax.tick_params(axis="x", colors=INK2, labelsize=10, length=0)
        ax.grid(axis="x", visible=False)
        ax.set_xlim(-0.5, 3.6)
        ax.set_ylim(0, top)
    runs = len(q["per_run"][0])
    title(fig, "Experiment 2: the homeostatic agent pushes the NPC onto fire less, but pushes less overall too",
          f"Final networks tested with the NPC on the agent's own grid ({q['layouts']} layouts). {runs} runs per group. "
          "Filled: trained with the NPC (out of reach); hollow: without.\nA push onto fire also leaves the agent next "
          "to fire, so this can be self-protection; the matched floor-push test that would tell them apart is not "
          "built.")
    save(fig, "results_3_exp2_pushes.png")


if __name__ == "__main__":
    experiment_1_reliability()
    experiment_1_survival()
    experiment_2_pushes()
