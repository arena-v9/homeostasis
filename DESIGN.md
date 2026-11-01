# Homeostasis Gridworld: Current Design

The living record of the experiment's design. Update it whenever the design changes. Code is the source of truth for
numbers (`gridworld_config.py`); this file explains what the design is and why.

Last updated: 2026-11-01.

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
| Map | 7×7 grid with a wall border (5×5 playable). Each episode places 2 fire tiles and 2 food items at random on floor, checking that food is reachable without crossing fire |
| Food | Eaten items respawn 10 steps later on a random empty floor tile |
| Fire | Hurts any body standing on it: −0.3 health per step |
| Actions | Up, down, left, right, stay. Moving into the NPC pushes it one square and the agent takes its place; a push into a wall fails |
| Episode | 100 steps, then truncated (not terminated: values bootstrap through the cutoff). Ends early if the agent dies |

Order of one step: the agent moves (and may push), the NPC moves, food respawns, bodies update, reward is computed.

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

The NPC never enters the agent's reward unless coupling is turned on (below).

## Observation

The agent sees the whole 7×7 map as 9 channels: wall, food, fire, spikes, cold, body, self, health, energy.

Both bodies appear on the same `body`, `health` and `energy` channels. Only `self` tells them apart. This shared
representation is the mechanism under test: a convolutional feature learned from the agent's own body ("a body next
to fire is about to lose health") can also respond at the NPC's square. `spikes` and `cold` are reserved for later
hazards and are currently all zeros.

## Agent and training

- DQN (from ARENA chapter 2) with a CNN Q-network: two 3×3 convolutions with 32 filters (padded), then linear layers
  1568 → 128 → 5.
- Replay buffer is circular; epsilon-greedy exploration; target network.
- During training, a push that would put the NPC on a tile that hurts it is discarded and the episode restarts, so
  the agent never experiences what such a push does and any avoidance comes from generalization.

## Experimental conditions

| Condition | Options |
|---|---|
| NPC state | Visible (NPC health and energy on the shared channels) or hidden (0 at the NPC's square) |
| Witnessing | The NPC never steps on fire, or sometimes does (chance 0.2 when next to fire), so the agent sees harm to the NPC that costs it nothing |
| Coupling | Weight w on the NPC's drive in the agent's drive: D = D_agent + w · D_npc. w = 0 is the main test; w = 0.25 and 0.5 are positive controls |

**Why coupling is included:** Yoshida & Man (2024) found that homeostatic agents who could observe a partner's
internal state ("cognitive empathy") but were rewarded only for their own homeostasis did not help the partner.
Prosocial behavior appeared only with a coupled drive ("affective empathy", w = 0.5). Their test was costly helping
with the partner's state as a separate input, not zero-cost harm avoidance through shared channels, so our question
is still open. But their result predicts a weak or null effect at w = 0, and the coupled conditions make a null
interpretable.

## Evaluation (planned)

Matched test layouts where pushing is free: a fire push (the NPC lands on fire) and a floor push (same geometry, NPC
lands on floor, agent ends equally close to fire). Measures: push rate (primary) and the Q-value gap, Q(push) minus
the best alternative (secondary, more sensitive). Avoidance = floor-push value − fire-push value.

## Status

| Part | State |
|---|---|
| Mechanics, observation, tests (`gridworld.py`, `test_gridworld.py`) | Done |
| Random layouts (`gridworld_layout.py`), batched env, CNN, solo training script | Done |
| Run outputs: one JSON folder per run, viewer page with run picker, curves and replays (`run_outputs.py`, `viewer.html`) | Done |
| Scripted NPC, witnessed condition, verification counters | Not started |
| Evaluation harness | Not started |

## Open issues

- **Training stability.** Under the PDF's original reward (−200 on death), two solo runs with identical settings
  reached 83% and 17% survival. Under the current design, one solo run (seed 1, 200k steps) passed 95% survival by
  about 60k environment steps and finished at 100%, with one dip to 84%. Not yet checked across seeds; Huber loss
  and gradient clipping remain the fallback if other seeds are unstable.
- **Tuning.** β = 10 and starvation damage 0.05 are judgment calls; check against solo-agent behavior.
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
