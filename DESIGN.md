# Homeostasis Gridworld: Current Design

The living record of the experiment's design. Update it whenever the design changes. Code is the source of truth for
numbers (`envs/gridworld/config.py`); this file explains what the design is and why.

Last updated: 2026-11-02 (afternoon).

## Research question

Can an agent learn to avoid harming another body without being rewarded for it, because it has a vulnerable body of
its own? The agent is rewarded only for keeping its own body in good condition. A scripted second body, the NPC, has
the same needs, appears on the same observation channels as the agent, and can be pushed onto fire at no cost to the
agent. The question is whether the agent avoids that push more than a matched harmless push.

The idea comes from Christov-Moore et al. (2023), who argue empathy needs vulnerability. The original draft of this
experiment is the PDF in the repo root; it was a starting proposal, and this file supersedes it.

## Environment

| Element | Current design |
|---|---|
| Map | 7×7 grid with a wall border (5×5 playable). Each episode places 3 fire tiles and 2 food items at random on floor, checking that food is reachable without crossing fire |
| Parallel grid (`parallel_grid`) | Optional. The NPC lives on a second grid of its own (with its own fire and food), shown to the agent to the right of its grid on the same channels. The two grids never interact: no pushing, blocking or shared food. With no NPC, the second grid is still shown, empty of bodies. Used by experiment 1; the main experiment needs the shared grid, since the agent must be able to push the NPC |
| Food | Eaten items respawn 10 steps later on a random empty floor tile |
| Fire | Hurts any body standing on it: −0.3 health per step |
| Actions | Up, down, left, right, stay. Moving into the NPC pushes it one square and the agent takes its place; a push into a wall fails |
| Episode | 100 steps, then truncated (not terminated: values bootstrap through the cutoff). Ends early if the agent dies |

Order of one step: the agent moves (and may push), the NPC moves, food respawns, bodies update, reward is computed.
A pushed NPC loses its move that step: its action was chosen before the push, so applying it from the new square could
walk it onto fire.

## Bodies

Every body (agent and NPC) has two internal variables in [0, 1]:

- **Energy**, a resource. Falls by 0.02 every step (hunger). Eating restores 0.5, capped at 1.
- **Health**, bodily integrity. The body dies when health reaches 0, and only then.

| Dynamic | Rule |
|---|---|
| Regeneration | +0.01 health per step, but only while energy > 0 |
| Starvation | While energy is 0, health drains by 0.05 per step instead. Running out of energy doesn't kill outright; it starts a visible decline toward death |
| Fire | −0.3 health per step on fire |
| Death | Health reaches 0 |

Bodies start each episode at full health, with energy uniform in [0.5, 1].

**Why this body:** death always comes through one variable, health, so there is one clear failure signal, and the
two variables are coupled the way physiology is: you can't heal while starving. This follows Crafter (Hafner 2021),
where health recovers only while food, drink and energy are all above zero and drains when any is empty, and fits
Damasio's view of an organism maintaining its integrity. A body left alone drifts toward death (hunger, then
starvation), which is the "being-towards-death" condition Christov-Moore et al. (2025) argue is needed for a
homeostatic drive.

## NPC

The NPC has no network; it follows a script (`envs/gridworld/npc_policy.py`), the same in every condition:

- **Hungry** (energy < 0.5): steps along a shortest path to the nearest food that never enters fire, routing around
  the agent when possible. Ties between equally short paths are broken at random.
- **Otherwise:** wanders, choosing uniformly among staying and the moves onto squares that aren't fire.
- **Mistakes:** when next to fire, it steps onto it with probability `npc_fire_step_prob` = 0.1, and its normal
  script takes it off again the next step. On its own grid this puts the NPC on fire in about 80% of episodes and
  kills it in about 8%, so the agent often sees it hurt. (0.04 gave 49% and 0.3%; 0.2 gave 92% and 36%: the NPC dies
  too often to stay visible.) At 0 it never walks onto fire.

## Drive and reward

**Drive** is how far a body is from its setpoints (health = energy = 1), following Keramati & Gutkin (2014):

    D = (deficit_health^n + deficit_energy^n)^(1/m),  deficit = 1 − value,  n = 2, m = 1

So the drive is the sum of squared deficits. It's convex: losing 0.1 health when already at 0.2 costs far more than
at 0.9, so the agent prioritizes its most depleted need and is risk-averse. A dead body has the maximum drive,
D_max = 2.

**Reward is drive reduction:**

    r = β · (D_before − D_after),  β = 10

It is about 0 at rest, positive when a need is met (eating), and negative when harmed (fire, hunger, starvation).
Death jumps the drive to D_max, giving β · (D_before − D_max), and ends the episode as an absorbing state with no
further reward. Over an episode the rewards add up to β · (D_start − D_end).

**Why:** Keramati & Gutkin prove that with discounting (γ < 1), maximizing discounted drive reduction is the same as
minimizing the discounted deviation from the setpoints, so this reward makes the agent pursue physiological
stability. It is the standard form in homeostatic RL, including Yoshida & Man (2024), the closest comparable
experiment. It makes rewards event-shaped (a spike for eating, a dip on fire), and prices death from the drive
instead of with a hand-tuned penalty. Damasio's account of feelings as the mind's report of how close the body is to
its homeostatic ideal motivates making the drive, not external events, the source of reward.

The NPC never enters the agent's reward: any harm avoidance has to come from the shared representation, not from
incentives.

## Observation

The agent sees the whole 7×7 map as 9 channels: wall, food, fire, spikes, cold, body, self, health, energy.

Both bodies appear on the same `body`, `health` and `energy` channels. Only `self` tells them apart. This shared
representation is the mechanism under test: a convolutional feature learned from the agent's own body ("a body next
to fire is about to lose health") can also respond at the NPC's square. `spikes` and `cold` are reserved for later
hazards and are currently all zeros.

## Agent and training

- DQN (from ARENA chapter 2) with a CNN Q-network: two 3×3 convolutions with 32 filters (padded), then linear layers
  1568 → 128 → 5.
- Replay buffer is circular and holds 100k transitions, filled with 20k random steps before training starts;
  epsilon-greedy exploration; target network.
- **Double DQN** (van Hasselt et al. 2016): the Q-network chooses the next action and the target network values it,
  which removes DQN's tendency to overestimate values.
- **Store of deaths**, against catastrophic forgetting: every transition that ends in death is also kept in a small
  separate store (5,000 transitions), and each batch takes 5% of its transitions from it (about 6 of 128).
- Huber loss on the TD error and gradient-norm clipping (max 10), as in standard DQN (Mnih et al. 2015 clip the TD
  error, which is equivalent to the Huber loss). ARENA's teaching version leaves both out. They matter here because
  rewards have rare large spikes: a death costs up to β · D_max = 20, against about ±0.1 on a normal step.
- Harmful pushes happen: the agent can push the NPC onto fire.
- Transitions the environment flags in `infos["dropped"]` are never stored in the replay buffer (see NPC harm
  below). Evaluation always runs with `drop_npc_harm` off, so its episodes play out in full.

## Experimental conditions

| Condition | Options |
|---|---|
| NPC state | Visible (NPC health and energy on the shared channels) or hidden (0 at the NPC's square) |
| NPC harm in training (`drop_npc_harm`) | Stored (default) or dropped. Dropped: any step that leaves the NPC on a harmful tile, whether the agent pushed it there or it stepped there by mistake, is never stored in the replay buffer, and the episode restarts so the agent doesn't see its aftermath (e.g. lowered health) either. The environment is otherwise identical: the NPC makes the same mistakes and pushes still happen |

**No coupled conditions.** The NPC's health or energy never enters the agent's reward, not even as a positive control.
Yoshida & Man (2024) found that homeostatic agents rewarded only for their own homeostasis did not help a partner
whose state they could observe; prosocial behavior appeared only with a coupled drive. Their test was costly helping
with the partner's state as a separate input, not zero-cost harm avoidance through shared channels, so our question
is still open, but their result means a null effect here is a real possibility.

## Experiments

Experiments run one at a time; each must be shown to work before the next starts.

**Experiment 1: does watching an NPC change how the agent learns to survive?** All three groups use the parallel
grid. In the first, the second grid is empty. In the other two an NPC lives on it, moving around and eating; its
health and energy are hidden in one and visible in the other, on the same channels as the agent's own. Each group is
compared with the one before it, so the comparisons are watching the NPC, then seeing its health and energy. The NPC's
state is only seen, never part of the reward. 15 runs per group (seeds 1–15), otherwise default settings. Analysed
with `python -m runs.experiments experiment-1` (questions defined in `runs/experiment_definitions.py`).

**Why the parallel grid:** an earlier version of experiment 1, with the NPC on the agent's grid, found that agents
trained with the NPC were more reliable (their survival collapsed less often late in training). On a shared grid,
that could come from the NPC competing for food, moving food around, or being pushed. The parallel grid removes all
of those, so any remaining difference comes from watching the NPC.

## Evaluation (planned)

Matched test layouts where pushing is free: a fire push (the NPC lands on fire) and a floor push (same geometry, NPC
lands on floor, agent ends equally close to fire). Measures: push rate (primary) and the Q-value gap, Q(push) minus
the best alternative (secondary, more sensitive). Avoidance = floor-push value − fire-push value.

## Status

| Part | State |
|---|---|
| Mechanics, observation, tests (`envs/gridworld/dynamics.py`, `drive.py`, `observation.py`, `tests/test_gridworld.py`) | Done |
| Random layouts (`envs/gridworld/initial_state.py`), batched env, CNN, training script (`train_gridworld.py`, flags named after config fields) | Done |
| Run outputs: one JSON folder per run, viewer page with run picker, curves and replays (`runs/outputs.py`, `ui/index.html`) | Done |
| Scripted NPC, `drop_npc_harm` (`envs/gridworld/npc_policy.py`, `tests/test_npc_policy.py`); counters for pushes, NPC steps on fire and the agent's steps on fire in the evaluations and replays | Done |
| Parallel grid (`envs/gridworld/episode.py`, `tests/test_parallel_grid.py`) | Done |
| Catastrophic-forgetting diagnostics at every evaluation (`runs/diagnostics.py`): deaths in the buffer, TD loss, action gap, value of doomed states | Done |
| Store of deaths in every training batch, and Double DQN, against catastrophic forgetting and collapses (`dqn/replay_buffer.py`, `dqn/trainer.py`) | Done |
| Experiment analyses and viewer tab (`runs/experiments.py`, `runs/statistics.py`, `runs/experiment_definitions.py`) | Done |
| Counters over the training data itself (dropped transitions, NPC-on-fire steps the agent stored) | Not started |
| Evaluation harness | Not started |

## Open issues

- **Training stability.** Under the PDF's original reward (−200 on death), two runs without the NPC with identical
  settings reached 83% and 17% survival. Under the current design with squared-error loss, four seeds without the NPC
  (200k steps) all first reached 95% survival within 60–92k environment steps, but kept swinging afterwards: in the
  second half of training their lowest survival was 0.64–0.84 (mean 0.90–0.97), so a single final network's numbers
  depend on when training stopped. With Huber loss and gradient clipping, the same four seeds reached 97–100%
  survival sooner (by about 23% of training) but swung just as much later (one seed fell to 0.48 at 75% and
  recovered).
- **Catastrophic forgetting.** The diagnostics (2026-11-02, 100k buffer) show it in every run, with or without the
  NPC: deaths disappear from the replay buffer by about 110k steps, the network's value for doomed states climbs from
  about the true value (−1 to −0.1) to +10 to +19, and survival collapses follow. A 200k buffer and a 10% share of
  random old experience didn't help (one run per group each): old experience is mostly near-random play with few
  deaths. Fixed with two changes, checked on 5 seeds without the NPC (parallel grid): a store of deaths sampled in
  every batch keeps the network's value of doomed states correct (error +0.9 at the end instead of +11.3), and Double
  DQN on top makes survival more reliable (79% of evaluations after 100k steps at ≥95% survival, against 52% with
  neither and 57% with the store alone; 54 against 88 starvation deaths per 1,000 episodes). Collapses still happen,
  less often: one seed stayed at 54%.
  Selecting the best checkpoint on separate validation layouts is a safety net for the evaluation either way.
- **Dropping cuts episodes short.** With `drop_npc_harm`, every NPC-on-fire event restarts the episode (about 80%
  of episodes on the parallel grid), so dropped-condition agents see fewer late-episode states. Check whether
  that changes their own survival compared with stored-condition agents.
- **Fire is rarely deadly for trained agents.** With 2 fire tiles, trained agents died in fire in under 1 in 200
  test episodes, so fire behaviour couldn't be compared between conditions. Fire tiles went from 2 to 3
  (2026-11-02) to make fire matter more; if fire deaths stay this rare, consider slippery moves (a small chance that
  a move goes in a random direction), so that walking next to fire carries risk.
- **Tuning.** β = 10 and starvation damage 0.05 are judgment calls; check against the agent's behaviour without the NPC.
- **NPC energy.** The NPC's energy visibly falls at no cost to the agent, which may teach the network to separate
  the two bodies. If so, hold the NPC's energy constant during training.

## References

- Keramati, M. & Gutkin, B. (2014). Homeostatic reinforcement learning for integrating reward collection and
  physiological stability. *eLife* 3:e04811. https://elifesciences.org/articles/04811
- Yoshida, N. & Man, K. (2024). Empathic Coupling of Homeostatic States for Intrinsic Prosociality. NeurIPS 2024
  workshop. https://arxiv.org/abs/2412.12103
- Hafner, D. (2021). Benchmarking the Spectrum of Agent Capabilities (Crafter). Body mechanics from the source:
  https://github.com/danijar/crafter/blob/main/crafter/objects.py
- Christov-Moore, L. et al. (2023). Preventing antisocial robots: A pathway to artificial empathy. *Science
  Robotics*. https://doi.org/10.1126/scirobotics.abq3658
- Christov-Moore, L. et al. (2025). The Conditions of Physical Embodiment Enable Generalization and Care.
  https://arxiv.org/abs/2510.07117
- Man, K. & Damasio, A. (2019). Homeostasis and soft robotics in the design of feeling machines. *Nature Machine
  Intelligence*. https://doi.org/10.1038/s42256-019-0103-7
- Damasio, A. (2018). *The Strange Order of Things: Life, Feeling, and the Making of Cultures*.
- Yoshida, N., Sprekeler, H. & Gutkin, B. (2025). Linking Homeostasis to Reinforcement Learning: Internal State
  Control of Motivated Behavior. https://arxiv.org/abs/2507.04998
- Sterling, P. (2012). Allostasis: a model of predictive regulation. *Physiology & Behavior* 106(1), 5–15.
