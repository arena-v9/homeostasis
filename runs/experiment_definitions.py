"""
What each experiment compares and asks. An experiment compares two groups of training runs; a group is every saved
run with exactly its `config_changes` (GridworldConfig settings changed from the defaults).
The first group is the baseline: differences are the second group minus the first.

Each question has a kind, which runs/experiments.py knows how to answer from the runs:

  better            Does the second group do better? Reliability and deaths by cause, from training's evaluations.
  test_dependence   Is the second group's advantage learned, or does it depend on the test setting? Every final
                    network is tested in both groups' settings.
  explanations      Candidate explanations, written here by us: not computed.
"""

EXPERIMENTS = {
    "experiment-1": {
        "title": "Experiment 1: Does watching an NPC change how the agent learns to survive?",
        "runs_per_group": 15,  # the analysis refuses to run until every group has this many runs
        "setup": (
            "Every agent sees two grids side by side: its own, and a second one with its own fire and food that it "
            "can never enter. In one group an NPC lives on the second grid, moving around and eating; in the other "
            "the second grid is empty. The agent and the NPC never interact. The NPC's health and energy are hidden. "
            "Everything else uses the default settings, and every run trains for 200k environment steps with Double "
            "DQN and a store of past deaths in every batch (against catastrophic forgetting)."
        ),
        "groups": [
            {"label": "No NPC", "phrase": "without the NPC",
             "config_changes": {"parallel_grid": True, "npc_enabled": False}},
            {"label": "NPC", "phrase": "with the NPC",
             "config_changes": {"parallel_grid": True, "npc_state_visible": False}},
        ],
        "questions": [
            {"kind": "better", "title": "Does an agent that watches the NPC do better?"},
            {"kind": "test_dependence",
             "title": "Is the agent learning to survive, or does it depend on the NPC being there?"},
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
