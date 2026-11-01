"""
How the NPC chooses its action. The scripted NPC comes later; until then it stands still.
"""

from typing import Callable

import numpy as np

from envs.gridworld.dynamics import Action
from envs.gridworld.state import GridworldState

# Chooses the NPC's action from the current state
NpcPolicy = Callable[[GridworldState, np.random.Generator], Action]


def stationary_npc_policy(state: GridworldState, rng: np.random.Generator) -> Action:
    """The NPC never moves."""
    return Action.STAY
