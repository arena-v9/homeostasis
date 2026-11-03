"""
What each experiment compares and asks. An experiment compares groups of training runs; a group is every saved run
with exactly its `config_changes` (GridworldConfig settings changed from the defaults).
Each group after the first changes one thing from the group before it, named by its `change`, and is compared with
that group: differences are each group minus the one before it.

Each question has a kind, which runs/experiments.py knows how to answer from the runs:

  better            Does each change do better? Reliability and deaths by cause, from training's evaluations.
  test_dependence   Is each change's advantage learned, or does it depend on the test setting? Every final network is
                    tested in every group's setting.
  explanations      Candidate explanations, written here by us: not computed.
"""

EXPERIMENTS = {
    "experiment-1": {
        "title": "Experiment 1: Does watching an NPC change how the agent learns to survive?",
        "runs_per_group": 15,  # the analysis refuses to run until every group has this many runs
        "setup": (
            "Every agent sees two grids side by side: its own, and a second one with its own fire and food that it "
            "can never enter. In the first group the second grid is empty. In the other two an NPC lives on it, "
            "moving around and eating; its state (health and energy) is hidden in one and visible in the other, drawn "
            "on the same channels as the agent's own. The agent and the NPC never interact, and the NPC's state "
            "never enters the agent's reward. Everything else uses the default settings, and every run trains for "
            "200k environment steps with Double DQN and a store of past deaths in every batch (against catastrophic "
            "forgetting)."
        ),
        "groups": [
            {"label": "No NPC", "phrase": "without the NPC",
             "config_changes": {"parallel_grid": True, "npc_enabled": False}},
            {"label": "NPC, state hidden", "phrase": "with the NPC, state hidden", "change": "Watching the NPC",
             "config_changes": {"parallel_grid": True, "npc_state_visible": False}},
            {"label": "NPC, state visible", "phrase": "with the NPC, state visible", "change": "Seeing the NPC's state",
             "config_changes": {"parallel_grid": True}},
        ],
        "questions": [
            {"kind": "better", "title": "Does watching the NPC help the agent, and does seeing its state help more?"},
            {"kind": "test_dependence",
             "title": "Is the agent learning to survive, or does it depend on what it sees of the NPC?"},
            {"kind": "explanations", "title": "Why? Candidate explanations", "hypotheses": [
                {
                    "name": "Shared features",
                    "idea": (
                        "The NPC's body appears on the same channels as the agent's. Watching it get hungry, walk to "
                        "food and eat trains the convolutional features the agent also uses for its own body."
                    ),
                    "prediction": (
                        "The advantage needs the NPC on the shared body channel: an NPC drawn on a channel of its own "
                        "should not help."
                    ),
                    "tests": ["Train with the NPC on a separate channel (new setting)."],
                },
                {
                    "name": "Moving input",
                    "idea": (
                        "A moving NPC keeps the picture changing from step to step, which may stop a greedy agent from "
                        "getting stuck repeating the same moves. That would help it without it learning anything "
                        "about the NPC."
                    ),
                    "prediction": (
                        "The advantage shows only while the NPC is there (section 2), and an NPC that moves at random "
                        "without eating helps just as much."
                    ),
                    "tests": ["Train with an NPC that wanders at random and never eats (new setting)."],
                },
            ]},
        ],
    },
}
