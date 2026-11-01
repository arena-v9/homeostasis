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


@dataclass(frozen=True)
class GridworldConfig:
    # Map: a grid_size x grid_size grid with a wall border, so the playable area is (grid_size - 2) x (grid_size - 2)
    grid_size: int = 7
    num_fire: int = 2  # fire tiles placed at random each episode (see initial_state.py)
    npc_enabled: bool = True  # False = the agent is alone (for checking the environment and the agent's learning)
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

    # Reward = drive reduction (Keramati & Gutkin 2014): r = reward_scale * (D_before - D_after), from the agent's
    # drive (plus `npc_drive_weight` times the NPC's drive, if coupled). About 0 at rest, positive when a need is met,
    # negative when harmed. Death jumps the drive to D_max (r = reward_scale * (D_before - D_max)) and ends the
    # episode: an absorbing state with no further reward.
    reward_scale: float = 10.0

    # Coupling (Yoshida & Man 2024's "affective empathy"): the agent's drive includes w times the NPC's drive.
    # 0 = the agent's reward ignores the NPC entirely. A dead NPC counts as maximum drive.
    npc_drive_weight: float = 0.0

    # Episodes are cut off (truncated, not terminated) after this many steps
    max_steps: int = 100

    # Conditions
    npc_state_visible: bool = True  # False = "hidden": the health/energy channels show 0 at the NPC's square
    witnessed: bool = False  # True = the NPC sometimes steps onto fire during training (used by the NPC script)
    npc_fire_step_prob: float = 0.2  # witnessed only: chance the NPC steps onto adjacent fire

    # Training: if a push would put the NPC on a tile that hurts it, the transition is discarded and the episode
    # restarts, so the agent never experiences what such a push does. Set False to let the push happen.
    discard_harmful_pushes: bool = True

    def agent_damage(self, tile: Tile) -> float:
        """Health the agent loses per step standing on `tile`."""
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
