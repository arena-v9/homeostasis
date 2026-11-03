"""
Constants and condition flags for the homeostasis gridworld. Every number in the environment lives here, so each
experiment is one `GridworldConfig`. DESIGN.md explains the design and where each choice comes from.
"""

from dataclasses import asdict, dataclass, field
from enum import IntEnum


class Tile(IntEnum):
    """
    What a square of the map is. Food is not a tile: it's an item that sits on a floor tile (see GridworldState).
    """

    FLOOR = 0
    WALL = 1
    FIRE = 2
    SPIKES = 3  # later stage: hurts both bodies, but the NPC is never seen on it
    COLD = 4  # later stage: hurts only the NPC


AGENT_MODELS = ("homeostatic", "psychopath")  # what the agent is rewarded for: see reward.py


@dataclass(frozen=True)
class GridworldConfig:
    # Map: a grid_size x grid_size grid with a wall border, so the playable area is (grid_size - 2) x (grid_size - 2)
    grid_size: int = 7
    num_fire: int = 3  # fire tiles placed at random each episode (see initial_state.py)
    npc_enabled: bool = True  # False = the agent is alone (for checking the environment and the agent's learning)
    # True = the NPC lives on its own grid, shown to the agent next to its own; the two never interact (see episode.py).
    # With npc_enabled False the second grid is still there, with its own fire and food, but nobody in it.
    parallel_grid: bool = False
    # Parallel grid only. True = the NPC lives on the agent's grid instead, and the second grid stays empty: the agent
    # sees the same 7 x 14 picture it trained on, but can now push the NPC. A test setting, not for training.
    npc_on_agent_grid: bool = False
    start_energy_range: tuple[float, float] = (0.5, 1.0)  # each body's starting energy is uniform in this range

    # Bodies. Energy is a resource (0 to 1); health is bodily integrity (0 to 1), and a body dies when health hits 0.
    energy_decay: float = 0.02  # energy lost by each body every step
    food_energy: float = 0.5  # energy gained from eating one food item (capped at 1)
    health_regen: float = 0.01  # health regained per step while energy > 0 (capped at 1, applied before damage)
    starvation_damage: float = 0.05  # health lost per step while energy is 0

    # Food: eaten items respawn after `food_respawn_steps` steps on a random empty floor tile
    num_food: int = 2
    food_respawn_steps: int = 10

    # Hazards: health lost per step by a body standing on the tile, as (damage to agent, damage to NPC).
    # Stage 1 only has fire; spikes and cold get entries in later stages.
    hazard_damage: dict[Tile, tuple[float, float]] = field(default_factory=lambda: {Tile.FIRE: (0.3, 0.3)})

    # Drive: how far a body is from its setpoints (health = energy = 1), following Keramati & Gutkin (2014):
    #     D = (deficit_health ** n + deficit_energy ** n) ** (1 / m),   deficit = 1 - value
    # n > 1 makes the drive convex, so a deficit costs more the deeper it already is. A dead body has the maximum
    # drive, D_max = 2 ** (1 / m).
    drive_exponent: float = 2.0  # n
    drive_root: float = 1.0  # m

    # Homeostatic reward = drive reduction (Keramati & Gutkin 2014): r = reward_scale * (D_before - D_after), from the
    # agent's own drive only: the NPC never enters the reward. About 0 at rest, positive when a need is met, negative when harmed. Death jumps the drive to D_max (r = reward_scale * (D_before - D_max)) and ends the
    # episode: an absorbing state with no further reward.
    reward_scale: float = 10.0

    # The agent model (one of AGENT_MODELS; see reward.py). "homeostatic" is rewarded by the drive reduction above.
    # "psychopath" gets `food_reward` per food item it eats and nothing else, and its body can't be hurt: neither fire
    # nor starvation lowers its health, so it never dies.
    agent_model: str = "homeostatic"
    food_reward: float = 1.0

    # Episodes are cut off (truncated, not terminated) after this many steps
    max_steps: int = 100

    # NPC script (npc_policy.py): below this energy it heads for the nearest food, otherwise it wanders
    npc_hunger_threshold: float = 0.5

    # Conditions
    npc_state_visible: bool = True  # False = "hidden": the health/energy channels show 0 at the NPC's square
    # Chance per step that the NPC, when next to fire, steps onto it by mistake (0 = never). On its own grid (3 fire
    # tiles), 0.1 puts the NPC on fire in about 80% of episodes and kills it in about 8%; 0.04 gave 49% and 0.3%,
    # 0.2 gave 92% and 36%.
    npc_fire_step_prob: float = 0.1
    # True = any step that leaves the NPC on a tile that hurts it (pushed there or by its own mistake) is never stored
    # in the replay buffer, and the episode restarts, so the agent never learns from seeing the NPC hurt.
    drop_npc_harm: bool = False

    def __post_init__(self):
        if self.agent_model not in AGENT_MODELS:
            raise ValueError(f"agent_model must be one of {AGENT_MODELS}, got {self.agent_model!r}")
        if self.npc_on_agent_grid and not self.parallel_grid:
            raise ValueError("npc_on_agent_grid needs parallel_grid")

    @property
    def agent_vulnerable(self) -> bool:
        """Whether the agent's body can be hurt (by hazards and starvation) and die. The psychopath's can't."""
        return self.agent_model != "psychopath"

    def agent_damage(self, tile: Tile) -> float:
        """Health a vulnerable agent loses per step standing on `tile`."""
        return self.hazard_damage.get(Tile(tile), (0.0, 0.0))[0]

    def npc_damage(self, tile: Tile) -> float:
        """Health the NPC loses per step standing on `tile`."""
        return self.hazard_damage.get(Tile(tile), (0.0, 0.0))[1]

    def to_dict(self) -> dict:
        """The config as plain JSON-ready values (tiles in `hazard_damage` by name), e.g. for a run's run.json."""
        values = asdict(self)
        values["hazard_damage"] = {Tile(tile).name.lower(): list(damage) for tile, damage in self.hazard_damage.items()}
        values["start_energy_range"] = list(self.start_energy_range)
        return values

    def changes_from_defaults(self) -> dict:
        """The settings that differ from `GridworldConfig()`'s defaults, as JSON-ready values."""
        defaults = GridworldConfig().to_dict()
        return {name: value for name, value in self.to_dict().items() if value != defaults[name]}

    @property
    def max_drive(self) -> float:
        """The drive of a dead body (both deficits at 1)."""
        return 2.0 ** (1.0 / self.drive_root)
