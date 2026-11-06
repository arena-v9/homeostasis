"""
Slide figures of the gridworld's design (no results): the map and the push, the shared observation channels, and the
two experiments' settings. PNGs in presentation/out/. Colours are the viewer's (ui/index.html), so the slides match
what the replays show. The example state is the one in README.md and the design PDF.

    python presentation/make_design_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

OUT = Path(__file__).resolve().parent / "out"
# The viewer's palette (ui/index.html, light theme)
WALL, FLOOR, GRID, FIRE, FOOD = "#6e6e6e", "#ffffff", "#e1e1e1", "#ff7800", "#3cb44b"
AGENT, NPC, HEALTH, ENERGY = "#285adc", "#963cc8", "#dc2828", "#e6be00"
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.color": INK, "figure.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE})

# The README's example: agent (A) next to the NPC (N), fire (X) behind it, food (F)
EXAMPLE = ["#######",
           "#....F#",
           "#.X...#",
           "#ANX..#",
           "#.....#",
           "#F....#",
           "#######"]
AGENT_STATE, NPC_STATE = (0.9, 0.4), (0.7, 0.6)  # (health, energy), as in the design PDF


def parse(rows):
    tiles = {(r, c): ch for r, row in enumerate(rows) for c, ch in enumerate(row)}
    find = lambda ch: next((pos for pos, t in tiles.items() if t == ch), None)
    return tiles, find("A"), find("N")


def draw_grid(ax, rows, x0=0, agent=None, npc=None, npc_on_fire=False, label=None):
    """One 7x7 grid with its tiles and bodies, top-left at (x0, 0); y grows downward."""
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            color = WALL if ch == "#" else FIRE if ch == "X" else FLOOR
            ax.add_patch(Rectangle((x0 + c, r), 1, 1, facecolor=color, edgecolor=GRID, lw=1))
            if ch == "F":
                ax.add_patch(Rectangle((x0 + c + 0.3, r + 0.3), 0.4, 0.4, facecolor=FOOD, edgecolor="none"))
    for pos, color in ((agent, AGENT), (npc, NPC)):
        if pos is not None:
            ax.add_patch(Circle((x0 + pos[1] + 0.5, pos[0] + 0.5), 0.34, facecolor=color, edgecolor="white", lw=2))
    if npc is not None and npc_on_fire:
        ax.text(x0 + npc[1] + 0.5, npc[0] - 0.12, "ouch", ha="center", va="bottom", fontsize=9, color=FIRE,
                fontweight="bold")
    if label:
        ax.text(x0 + 3.5, 7.35, label, ha="center", va="top", fontsize=11, color=INK2)


def setup(ax, width, height=7.9):
    ax.set_xlim(-0.2, width + 0.2)
    ax.set_ylim(height, -0.6)
    ax.set_aspect("equal")
    ax.axis("off")


def legend(fig, y=0.06, extra=()):
    items = [(AGENT, "o", "agent (learns)"), (NPC, "o", "NPC (scripted, same body)"), (FIRE, "s", "fire (−0.3 health/step)"),
             (FOOD, "s", "food (+0.5 energy)"), (WALL, "s", "wall"), *extra]
    handles = [plt.Line2D([], [], marker=m, color="none", markerfacecolor=c, markersize=11, label=t) for c, m, t in items]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y - 0.04), ncol=len(items), frameon=False,
               fontsize=10)


def title(fig, text, sub):
    fig.text(0.04, 0.95, text, fontsize=15, fontweight="bold", va="top")
    fig.text(0.04, 0.885, sub, fontsize=10.5, color=INK2, va="top")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)
    print("wrote", OUT / name)


def the_push():
    """Before and after: walking into the NPC pushes it one square, here onto fire."""
    _, agent, npc = parse(EXAMPLE)
    after = [row.replace("A", ".").replace("N", ".") for row in EXAMPLE]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.8, bottom=0.12)
    draw_grid(ax, [row.replace("A", ".").replace("N", ".") for row in EXAMPLE], 0, agent, npc, label="Before: the agent moves right")
    draw_grid(ax, after, 10, (agent[0], agent[1] + 1), (npc[0], npc[1] + 1), npc_on_fire=True,
              label="After: the NPC is on fire")
    ax.add_patch(FancyArrowPatch((7.4, 3.5), (9.6, 3.5), arrowstyle="-|>", mutation_scale=22, color=INK2, lw=2))
    ax.add_patch(FancyArrowPatch((agent[1] + 0.85, agent[0] + 0.5), (npc[1] + 0.2, npc[0] + 0.5), arrowstyle="-|>",
                                 mutation_scale=14, color=INK, lw=1.8))
    setup(ax, 17)
    legend(fig)
    title(fig, "Pushing the NPC onto fire costs the agent nothing",
          "Walking into the NPC pushes it one square and the agent takes its place. The agent's reward is only its "
          "own drive reduction:\nthe NPC's health never enters it.")
    save(fig, "design_1_push.png")


def channels():
    """The example state as the network sees it: the channels where the bodies appear."""
    tiles, agent, npc = parse(EXAMPLE)
    grids = {name: np.zeros((7, 7)) for name in ("body", "self", "health", "energy", "fire")}
    for (r, c), ch in tiles.items():
        grids["fire"][r, c] = ch == "X"
    for pos, (h, e), is_self in ((agent, AGENT_STATE, True), (npc, NPC_STATE, False)):
        grids["body"][pos] = 1
        grids["self"][pos] = is_self
        grids["health"][pos], grids["energy"][pos] = h, e
    shown = [("fire", FIRE, "fire"), ("body", INK2, "body: both bodies"), ("self", AGENT, "self: the agent only"),
             ("health", HEALTH, "health: both bodies"), ("energy", ENERGY, "energy: both bodies")]
    fig, axes = plt.subplots(1, 5, figsize=(12, 4.2))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.66, bottom=0.08, wspace=0.12)
    for ax, (name, color, label) in zip(axes, shown):
        g = grids[name]
        for r in range(7):
            for c in range(7):
                v = g[r, c]
                ax.add_patch(Rectangle((c, r), 1, 1, facecolor=color if v else FLOOR, alpha=0.25 + 0.75 * v if v else 1,
                                       edgecolor=GRID, lw=0.8))
                if v and name in ("health", "energy"):
                    ax.text(c + 0.5, r + 0.55, f"{v:.1f}", ha="center", va="center", fontsize=8.5, color=INK,
                            fontweight="bold")
        ax.set_title(label, fontsize=10.5, color=INK2 if name != "self" else AGENT,
                     fontweight="bold" if name == "self" else "normal")
        ax.set_xlim(0, 7); ax.set_ylim(7, 0); ax.set_aspect("equal"); ax.axis("off")
    title(fig, "Both bodies appear on the same channels; only 'self' tells them apart",
          "Five of the network's 9 input channels for the example state (agent: health 0.9, energy 0.4; NPC: health "
          "0.7, energy 0.6).\nA convolutional feature learned from the agent's own body can respond at the NPC's "
          "square: the shared representation under test.")
    save(fig, "design_2_shared_channels.png")


def experiments():
    """Experiment 1's parallel grid (training) and experiment 2's push test (the NPC moved onto the agent's grid)."""
    own = ["#######", "#....F#", "#.X...#", "#.....#", "#..X..#", "#F..X.#", "#######"]
    other = ["#######", "#F....#", "#...X.#", "#.X...#", "#.....#", "#..X.F#", "#######"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.7, bottom=0.14, wspace=0.08)
    ax = axes[0]
    draw_grid(ax, own, 0, agent=(3, 2))
    draw_grid(ax, other, 7.6, npc=(3, 4))
    ax.text(3.5, 7.35, "agent's grid", ha="center", va="top", fontsize=10, color=INK2)
    ax.text(11.1, 7.35, "NPC's grid (never entered)", ha="center", va="top", fontsize=10, color=INK2)
    ax.set_title("Training (experiments 1 and 2): the parallel grid", fontsize=11.5, loc="left")
    setup(ax, 14.6)
    ax = axes[1]
    test = ["#######", "#....F#", "#.X...#", "#.....#", "#..X..#", "#F..X.#", "#######"]
    draw_grid(ax, test, 0, agent=(4, 1), npc=(4, 2))  # fire at (4, 3): moving right pushes the NPC onto it
    draw_grid(ax, ["#######"] + ["#.....#"] * 5 + ["#######"], 7.6)
    ax.text(3.5, 7.35, "the NPC on the agent's grid", ha="center", va="top", fontsize=10, color=INK2)
    ax.text(11.1, 7.35, "second grid left empty", ha="center", va="top", fontsize=10, color=INK2)
    ax.set_title("Experiment 2's test: can it push the NPC onto fire?", fontsize=11.5, loc="left")
    setup(ax, 14.6)
    legend(fig, y=0.09)
    title(fig, "The agent watches the NPC on a grid it can never enter, then meets it",
          "Experiment 1: no NPC vs NPC with its health and energy hidden vs visible. Experiment 2: a psychopath agent "
          "(food reward only,\ncannot be hurt) vs the homeostatic agent, tested on pushes with the NPC moved onto its "
          "grid. Example layouts.")
    save(fig, "design_3_experiments.png")


if __name__ == "__main__":
    the_push()
    channels()
    experiments()
