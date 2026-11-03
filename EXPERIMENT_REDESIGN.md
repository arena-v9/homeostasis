# Next Experiments

For Amanda and Fernando · 3 November 2026 · a 10-minute read. Nothing here has been run yet. Results so far are in [RESULTS.md](RESULTS.md); the literature behind the plan is in [PROJECT_WALKTHROUGH.md](PROJECT_WALKTHROUGH.md), Section 9.

## What we're testing

The claim we're building, in two parts:

1. **Not harming others.** An agent rewarded only for its own body avoids pushing another body onto fire because
   - (a) it can be hurt by fire itself, and
   - (b) its network sees the other body with the same features it uses for its own.

   This avoidance may fade once the agent learns that harming costs it nothing.
2. **Learning danger from others.** An agent can learn that something is dangerous by watching it hurt others, but only if it has a reason to model their harm (the auxiliary head) and a body of its own that can be hurt.

**Why redesign:** Experiment 2 changed four things at once. The psychopath differs in reward, in whether fire hurts it, in whether it can die, and in appetite. So it can't tell us which one matters. From now on, **each experiment changes one thing** from a reference agent.

**Reference agent (A0):** Experiment 1's "NPC, state visible" group (`--parallel-grid true`, seeds 1–15). It is homeostatic and can be hurt, it watches the NPC on the second grid, and it has no head. 15 runs already exist.

**Main measures** (from the existing harness, per run):
- **Push experiments:** NPC-specific push avoidance, meaning push avoidance minus push-empty avoidance. It is about +5 points for A0.
- **TV experiments:** ball avoidance, meaning steps beside the ball minus steps onto it.

## The experiments, in order

| # | What changes | Tests | Expected if we're right | Runs | New code |
|---|---|---|---|---|---|
| D1 | A new harness scene, no training | Is push avoidance the network mixing up the NPC's body with its own? | Tells mix-up from consequence | 0 (75 networks) | Harness test |
| D3 | Two new harness scenes, no training | Why does the psychopath push more after watching? | Tells "goes through bodies to food" from "follows bodies" | 0 (45 networks) | Harness tests |
| D4 | A new harness scene, no training | Does a danger learned by watching carry over to how it treats others? | Violent TV + head avoids pushing onto the ball | 0 (20 networks) | Harness test |
| B0 | 5 more seeds for Experiment 3 | Does the +13 ball result hold? | Yes | 20 | None |
| A1 | Fire doesn't hurt the agent | Does avoidance need the agent's own vulnerability? | ≈ 0 avoidance | 15 | Small flag |
| B2 | Psychopath watches the violent TV, with head | Is knowing the ball hurts bodies enough without caring? | No ball avoidance | 5 | None, plus one head readout |
| A2 | NPC on its own channel | Does avoidance need the NPC to look like the agent? | ≈ 0 avoidance | 15 | Medium |
| B1 | Head predicts energy, not health | Is it predicting harm specifically that matters? | No ball avoidance | 10 | Small |
| A4 | NPC within reach during training | Does avoidance survive learning that pushing is free? | Shrinks over training | 15 | Small (checkpoints) |

That's 80 training runs, about 1 hour of GPU at 10 at a time. The D tests need no training.

---

## D1 · Mix-up or consequence?

**Setup:**
- A new harness scene: the NPC stands with fire right beside it, and the agent is at least 2 squares away, so no move pushes anything.
- Read Q(move towards the fire's side) with the fire on that side, against a control with the fire elsewhere next to the NPC.
- Run on the 45 Experiment 1 networks and the 30 psychopaths.

```text
Push (exists)            Distant body (new)
. . . . .                . . f . .
. . f . .                . . N F .
A→N F . .                . . . . .
. . . . .                . . . . .
. . . . .                A→. . . .
A agent, N NPC, F fire (test), f fire (control)
```

**Tests:** whether the ~5 points of NPC-specific push avoidance come from the agent anticipating the NPC's burn, or from its network reading the NPC's body as its own. Both bodies are on the same `body` channel; only `self` tells them apart.

**Reading the result:**
- **Q(right) drops in the distant scene:** a mix-up. The agent avoids fire next to *a body*, whoever's it is.
- **It only drops when the move would push the NPC:** the agent is anticipating the consequence of the push.

Either is a result; they lead to different papers.

## D3 · Why does the psychopath push more after watching?

**What we know:** the psychopath trained with the NPC pushes more (1.81 vs 1.20 per episode). It isn't because it meets the NPC more: both get the same number of push chances. When it can push, it does so more often (18.8% vs 13.3% of floor-push chances), onto fire and floor alike.

**Setup:** two pairs of new harness scenes, run on both psychopath groups and the homeostatic NPC group.
- **In the way:** the NPC stands between the agent and food (food straight behind it), against the same scene with the food off to one side. Measure how often the agent walks into the NPC.
- **No food:** the NPC 2–3 squares away and no food on screen. Measure how often the move closes the distance, against a matched scene with an empty square there.

**Tests:** two explanations.
- **A more determined forager:** watching made it a better forager (15.3 vs 14.3 food per episode in training), so it goes through a body in its way.
- **It follows bodies:** on the parallel grid the NPC walks to food, so a food-seeker may learn "go where a body goes". On a shared grid, following means bumping.

**Expected:**
- **Extra pushing only in "in the way":** the forager explanation.
- **It also approaches with no food in view:** it follows bodies.

## D4 · Push the NPC onto the ball

**Setup:** a copy of the existing push test with the ball in place of fire, run on the 20 Experiment 3 networks.

**Tests:** the violent-TV agents with the head learned only from watching that the ball hurts bodies. Do they also avoid pushing another body onto it?

**Expected:** yes for violent TV + head only. That would be a danger learned from others and applied to others, which links both parts of the claim. It would be our strongest single result.

## B0 · Experiment 3, five more seeds

**Setup:** the same four groups, seeds 6–10:

```bash
python train_gridworld.py --tv neutral --aux-weight 0  --seed 6   # ... to 10
python train_gridworld.py --tv violent --aux-weight 0  --seed 6
python train_gridworld.py --tv neutral --aux-weight 10 --seed 6
python train_gridworld.py --tv violent --aux-weight 10 --seed 6
```

Then raise `runs_per_group` for experiment 3 to 10 in `runs/experiment_definitions.py`.

**Tests:** whether violent vs neutral with the head (+13 points, p = 0.008) holds. With 5 seeds, no within-group test can reach p < 0.05; with 10 it can.

## A1 · Fire-immune agent

**Setup:**
- A0's config (`--parallel-grid true`, seeds 1–15) plus one new flag that makes fire harmless to the agent only.
- The NPC is still hurt by fire.
- The agent still gets hungry, is rewarded by its own drive, and can starve and die.
- Today the damage lives in `hazard_damage` as (damage to agent, damage to NPC), with no flag. The new flag sets the agent's share to 0.

**Tests:** whether harm avoidance needs the agent to be vulnerable to this specific harm. Its reward, appetite and mortality are all unchanged. This is the clean version of the psychopath comparison.

**Expected:** NPC-specific push avoidance ≈ 0, like the psychopath.

**Note:** this agent will walk over fire freely, so it gets more push chances in live play. The matched harness corrects for that, which is why it's the main measure.

## B2 · Psychopath watching the violent TV, with the head

**Setup:**

```bash
python train_gridworld.py --tv violent --aux-weight 10 --agent-model psychopath --seed 1   # ... to 5
```

**Tests:** knowing vs caring. The head can learn that the ball hurts bodies, but the psychopath's reward doesn't depend on its health.

**Expected:** no ball avoidance.

**Also read the head:**
- what it predicts for **a TV body** on the ball (needs one new readout);
- what it predicts for **itself** on the ball.

It may learn "the ball hurts them, not me", which would be a clean self/other distinction.

## A2 · NPC on its own channel

**Setup:**
- A0's config plus a new observation option: the NPC is drawn on a channel of its own (with its own health and energy channels) instead of the shared `body` channel.
- The same option applies in the harness.
- Seeds 1–15.

**Tests:** whether avoidance needs the other body to look like the agent's own body to the network.

**Expected:** ≈ 0 avoidance. If avoidance survives, it doesn't come from shared features.

**Code:** medium. The number of input channels changes, and the harness and TV must follow.

## B1 · A head that predicts something other than harm

**Setup:** Experiment 3's head, but trained to predict each body's **energy** change instead of its health change. Violent and neutral TV, seeds 1–5 each.

**Tests:** whether the ball result comes from predicting harm specifically, or from any extra task that makes the network attend to bodies.

**Expected:** no ball avoidance.

## A4 · NPC within reach during training

**Setup:**
- The shared grid (`--parallel-grid false`, the default): the agent can push the NPC throughout training, and nothing that happens to the NPC affects the agent's reward.
- Save the network at 25/50/75/100% of training and run the harness on each.
- Seeds 1–15.

**Tests:** whether the avoidance survives experience. The agent's learning only sees its own reward. Once it has pushed the NPC onto fire a few times with no cost to itself, its values should correct, and the avoidance should fade.

**Expected:** NPC-specific avoidance shrinks across checkpoints. That would be a safety-relevant negative result: emergent "kindness" is trained away when harm is free.

**Caveat:** on a shared grid the NPC also competes for food (why Experiment 1 moved to the parallel grid). That's fine here, because the measure is harm, not reliability.

---

## Order

| When | Do | GPU |
|---|---|---|
| **Wed 4 Nov** | Build D1, D3, D4 and run them on existing networks. Launch B0. Write the fire-immune flag, the energy target and checkpoint saving. Launch A1 and B2 | 40 runs |
| **Thu 5 Nov** | Write the separate-channel option. Launch A2, B1 and A4. Analyse and make figures | 40 runs |
| **Fri 6 Nov** | Present. Anything unfinished goes on the "next steps" slide with its prediction | — |

## Rules for every experiment

- **Write the prediction down first:** before launching, record the main measure, the comparison and the expected direction as one line in `runs/experiment_definitions.py`.
- **15 seeds** for any main comparison (5 can't reach p < 0.05). B2 and B1 start at 5 as quick checks.
- **Freeze the harness:** don't change existing tests now. Build the new D tests on seeds 1–5, then check them on 6–15.
- **No NPC in the reward,** not even as a control. At most 10 runs on the GPU at once.

**Later, for the paper, not this week:** a price test (how much food the agent gives up to avoid harm) in a larger room; a dose-response for fire damage; a second network type; a second environment.
