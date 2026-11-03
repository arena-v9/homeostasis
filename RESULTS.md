# Results

The record of every experiment: setup, results, what they mean, and what's planned. Numbers come from the saved analyses (`ui/outputs/experiments/experiment-*.json`, made by `python -m runs.experiments experiment-N`) unless a line says otherwise. Each run is one trained network; "13/15" means 13 of 15 runs went the stated way.

Last updated: 3 November 2026.

## How to read the measures

- **Reliability:** the share of a run's evaluations after 100k environment steps in which the agent survived at least 95% of 64 test episodes. It measures how stably the agent learned to survive.
- **Harness avoidance** (`runs/harness.py`): matched pairs of hand-built scenes that differ in one tile, 256 pairs per test. Avoidance is how much less often the agent takes a move when it causes harm (control rate minus test rate), in percentage points.
  - **push:** the move pushes the NPC onto fire, against onto floor.
  - **push-empty:** the same scenes without the NPC.
  - **NPC-specific push avoidance** = push − push-empty, the part due to the NPC rather than the agent's own position.
  - **ball:** stepping onto the ball, against beside it.
  - **own fire:** stepping into fire, against beside it (sanity check).
- **Live push test:** pushes per episode with the NPC moved onto the agent's grid at test time (512 layouts). It mixes how often a push chance comes up with what the agent chooses, so the matched harness is the main measure.

## Index

| Experiment | Question | Runs | Status | Headline |
|---|---|---|---|---|
| Pilot | Does an NPC on the agent's grid change learning? | — | Superseded | More reliable with the NPC, but confounded; not comparable |
| 1 | Does watching an NPC change how the agent learns? | 45 | Done | Watching makes learning more reliable (75% → 91%); seeing its health adds nothing |
| 2 | Does an agent that can't be hurt harm more? | 30 + 15 reused | Done | Homeostatic agent avoids pushing the NPC onto fire; psychopath doesn't |
| 3 | Can the agent learn a danger by watching it on TV? | 20 | Done, 5 seeds | Only with the auxiliary head: avoids the ball (+13 vs neutral TV) |
| Next | One change at a time (see [EXPERIMENT_REDESIGN.md](EXPERIMENT_REDESIGN.md)) | 80 planned | Planned | Predictions recorded below |

---

## Pilot (superseded): NPC on the agent's own grid

- **Setup:** an earlier version of Experiment 1, with the NPC on the agent's grid, using older training code (before the store of deaths and Double DQN).
- **Result:** agents trained with the NPC were more reliable: 86% of late evaluations reached 95% survival, against 62% without it. An intermediate parallel-grid set, also without the forgetting fixes, gave 61% vs 72% (p = 0.11).
- **Interpretation:** on a shared grid the NPC also competes for food, moves food around and can be pushed, so the cause was unclear. This is why Experiment 1 moved to the parallel grid.
- **Do not pool or compare** these numbers with the experiments below; the training code differs.

## Experiment 1: does watching an NPC change how the agent learns?

**Setup.**
- Parallel grid: the agent sees its own grid plus a second grid it can never enter. Three groups, seeds 1–15, 200k steps.

  | Group | Flags | Second grid |
  |---|---|---|
  | No NPC | `--parallel-grid true --npc-enabled false` | Empty |
  | NPC, state hidden | `--parallel-grid true --npc-state-visible false` | NPC shown, health and energy hidden |
  | NPC, state visible | `--parallel-grid true` | NPC shown with health and energy |

- The NPC never enters the agent's reward.
- The harness was run afterwards on the final networks, with the NPC placed on the agent's grid, which none of them had seen.

**Results: learning.**

| | No NPC | NPC, hidden | NPC, visible |
|---|---|---|---|
| Reliability | 75% | 91% | 93% |
| Starvation deaths per 1,000 test episodes | 46 | 25 | 14 |

- Watching (hidden vs no NPC): reliability +15 points (95% interval +7 to +24, p = 0.003); starvation deaths −22 (p = 0.016).
- Seeing its state (visible vs hidden): reliability +3 points (p = 0.55), no clear difference.
- **Learned, not just present at test:** final networks tested without the NPC lose less than 0.02 survival. But the final networks alone can't show the gain was learned rather than a training-time effect.

**Results: harm, from the harness.**

| | No NPC | NPC, hidden | NPC, visible |
|---|---|---|---|
| Push avoidance | +6 (13/15) | +14 (14/15) | +8 (13/15) |
| Push-empty avoidance | +3 | +3 | +3 |
| **NPC-specific** | **+3 (9/15)** | **+11 (14/15)** | **+5 (10/15)** |
| Ball avoidance | −1 | −1 | 0 |
| Own fire (sanity) | +16 | +17 | +18 |

- Watching vs no NPC: push avoidance +8 (95% interval +3 to +13, p = 0.005).
- Visible vs hidden: −6 (p = 0.06), so seeing the NPC's health doesn't help.

**Interpretation.**
- Watching another body makes learning to survive more reliable. **Why is open:**
  - **Shared features:** watching the NPC trains filters the agent also uses for itself.
  - **Moving input:** a moving thing on screen acts like augmentation, with nothing social about it.
  - Not yet tested: the random-walker and separate-channel controls.
- These agents, never rewarded for anything about the NPC, push it onto fire less often than onto floor. Seeing its health adds nothing. This fits the agent's own fear of fire responding to a body drawn on the shared `body` channel. We can't yet tell whether that is transfer or a mix-up of the NPC with itself; see D1 and A2 below.
- No group avoids the ball: plain DQN learns nothing about a hazard it never met.

## Experiment 2 (Fernando): does an agent that can't be hurt harm more?

**Setup.**
- Parallel grid, NPC state visible, seeds 1–15. The psychopath is rewarded +1 per food item, and fire and starvation can't hurt it, so it never dies.

  | Group | Flags |
  |---|---|
  | Psychopath, no NPC | `--parallel-grid true --npc-enabled false --agent-model psychopath` |
  | Psychopath, NPC | `--parallel-grid true --agent-model psychopath` |
  | Homeostatic, NPC | Experiment 1's visible group, reused |

- Test: the NPC is moved onto the agent's grid (`npc_on_agent_grid`), on 512 layouts, plus the harness.

**Results: live push test** (per episode).

| | Psychopath, no NPC | Psychopath, NPC | Homeostatic, NPC |
|---|---|---|---|
| Pushes | 1.20 | 1.81 | 1.04 |
| Pushes onto fire | 0.11 | 0.18 | 0.05 |

- Watching made the psychopath push more: +0.62 pushes (p = 0.002) and +0.07 onto fire (p = 0.004).
- Homeostatic vs psychopath: −0.77 pushes (p = 0.004) and −0.13 onto fire (p < 0.001).

**Results: per push chance** (a one-off analysis of the live push test; not yet part of `runs/experiments.py`).

| | Takes a push onto fire | Takes a push onto floor | Avoidance | Runs > 0 |
|---|---|---|---|---|
| Psychopath, no NPC | 10.5% | 13.3% | +2.8 | 14/15 |
| Psychopath, NPC | 16.3% | 18.8% | +2.4 | 13/15 |
| Homeostatic, NPC visible | 5.7% | 14.2% | +8.5 | 15/15 |
| Homeostatic, NPC hidden | 6.7% | 14.2% | +7.5 | 15/15 |
| Homeostatic, no NPC | 5.1% | 13.0% | +7.9 | 15/15 |

The psychopath meets about 1.27× as many push chances, but takes each fire chance about 3× as often.

**Results: harness.**

| | Psychopath, no NPC | Psychopath, NPC | Homeostatic, NPC |
|---|---|---|---|
| Pushes onto fire vs onto floor | 18% vs 18% | 20% vs 20% | 15% vs 24% |
| NPC-specific avoidance | 0 | 0 | +5 (10/15) |
| Own fire (sanity) | 0 | 0 | +18 |

Homeostatic vs psychopath: push avoidance +8 (95% interval +5 to +13, p < 0.001).

**Interpretation.**
- A vulnerable agent avoids pushing a body onto fire; an invulnerable one doesn't distinguish fire from floor (as it doesn't for itself). This holds in matched scenes and per push chance, in every homeostatic run.
- **Which difference causes it is open.** The psychopath differs in reward, vulnerability, mortality and appetite at once. A1 (a fire-immune homeostatic agent) isolates vulnerability.
- Watching makes the psychopath push more in general, but not more selectively onto fire.
  - It meets the NPC equally often (548 vs 547 fire-push chances, 4,447 vs 4,241 floor-push chances per 512 episodes).
  - When it can push, it does so more often (18.8% vs 13.3% of floor chances).
  - Two candidate reasons: watching made it a more determined forager (15.3 vs 14.3 food per episode in training, from each run's `run.json`), so it goes through a body in its way; or it learned to follow bodies.
  - D3 tells them apart.
- Watching doesn't matter for the homeostatic agent per push chance (+7.9 without an NPC, +7.5 to +8.5 with). It does matter in the harness (+3 vs +11 NPC-specific). The two measures disagree, so **"watching helps harm avoidance" is not established.**
- Live counts alone mislead; report the matched or per-chance measures.

## Experiment 3: can the agent learn a danger by watching it on TV?

**Setup.**
- The agent is alone on its grid and watches a TV of recorded clips: a pusher drags a victim towards fire or a ball.
  - **Violent TV:** the drag ends on the fire or the ball, and the ball hurts like fire.
  - **Neutral TV:** the drag stops one square short, and the ball is harmless.
- Health and energy are shown on the TV. The agent never meets the ball or another body on its own grid in training.
- Half the agents have an **auxiliary head** (weight 10) that predicts every body's health change. It is not a reward.
- Seeds 1–5:

  | | No head | With head |
  |---|---|---|
  | Neutral TV | `--tv neutral` | `--tv neutral --aux-weight 10` |
  | Violent TV | `--tv violent` | `--tv violent --aux-weight 10` |

**Results.**

| | Neutral TV | Violent TV | Neutral TV + head | Violent TV + head |
|---|---|---|---|---|
| Ball avoidance | +3 (4/5) | +2 (3/5) | −3 (1/5) | **+10 (5/5)** |
| Steps onto ball vs beside it | 16% vs 19% | 19% vs 21% | 23% vs 20% | **11% vs 21%** |
| Head's predicted health change, agent on the ball | — | — | 0.00 | **−0.25** (fire: −0.29) |
| NPC-specific push avoidance | +11 (5/5) | +3 (3/5) | +3 (2/5) | +2 (3/5) |
| Reliability | 95% | 98% | 97% | 94% |

- **Violent vs neutral, with the head:** ball avoidance +13 (95% interval +9 to +17, p = 0.008).
- Violent vs neutral without the head: −1 (p = 0.57).
- Violent TV vs neutral TV lowered push avoidance (−11, p = 0.04), as did adding the head (−11, p = 0.04). Exploratory.
- No clear reliability differences.

**Interpretation.**
- Plain DQN learns nothing from watching harm. With a head that has to predict bodies' harm, the agent learns that the ball hurts bodies, applies that to its own body (−0.25 predicted, close to fire's −0.29), and avoids the ball. Observation alone is inert; a reason to model others' harm makes it count.
- **Caveats:**
  - Only 5 seeds: no within-group test can pass p < 0.05, so B0 adds seeds 6–10.
  - We don't yet know whether vulnerability is needed (B2) or whether predicting harm specifically matters (B1).
- The drop in push avoidance with the violent TV or the head is unexplained and could be noise at 5 seeds.

---

## Planned: predictions recorded before running

From [EXPERIMENT_REDESIGN.md](EXPERIMENT_REDESIGN.md). Each changes one thing from a reference:
- **A experiments:** Experiment 1's visible group, A0.
- **B experiments:** Experiment 3's violent TV + head.

Fill in Result and Verdict when each finishes; don't edit the prediction. Then update the experiment's entry in `PROJECT_EXPERIMENTS` in `ui/index.html` (status, page, result), so the viewer's Project map links to its page.

| ID | Change | Main measure | Prediction | Runs | Status | Result | Verdict |
|---|---|---|---|---|---|---|---|
| D1 | Distant-body scene, existing networks | Drop in Q(move towards the NPC's fire) when no push is possible | Decides mix-up vs consequence (no direction predicted) | 0 | Planned | | |
| D3 | "In the way" and "no food" scenes, existing networks | Walking into the NPC; moves that close the distance | Gap only "in the way" → more determined forager; gap also with no food → follows bodies | 0 | Planned | | |
| D4 | Push onto the ball, existing TV networks | Avoidance of pushing the NPC onto the ball | Violent + head > 0; others ≈ 0 | 0 | Planned | | |
| B0 | Experiment 3 seeds 6–10 | Ball avoidance, violent vs neutral with head | Stays positive, p < 0.05 within group | 20 | Planned | | |
| A1 | Fire doesn't hurt the agent | NPC-specific push avoidance | ≈ 0, below A0 | 15 | Planned | | |
| B2 | Psychopath watches violent TV, with head | Ball avoidance | ≈ 0 | 5 | Planned | | |
| A2 | NPC on its own channel | NPC-specific push avoidance | ≈ 0, below A0 | 15 | Planned | | |
| B1 | Head predicts energy, not health | Ball avoidance | ≈ 0 | 10 | Planned | | |
| A4 | NPC within reach in training, checkpoints | NPC-specific push avoidance over training | Falls from early to late checkpoints | 15 | Planned | | |

## Template for a new entry

```markdown
## Experiment N: <question>

**Setup.** What changes from which reference, flags, seeds, steps. What's held fixed.
**Prediction** (copied from the table above, written before running).
**Results.** One table; key comparisons with interval, p and runs > 0.
**Interpretation.** What it shows, what it doesn't, what to run next.
```
