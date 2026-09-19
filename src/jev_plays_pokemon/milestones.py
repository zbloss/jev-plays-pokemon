"""Deterministic tracker for the scripted current-objective milestone list.

This is the sole source of the navigation macro's destination (see
`CONTEXT.md`'s "Navigation macro" entry): a fixed, ordered list of known
Pokemon Red story beats, each checked against a game-state snapshot to
report which are completed / current / future. Purely scripted logic - no
AI component, no manual/chat intervention, no PyBoy or ROM dependency.

Takes exactly the two `GameState` fields (`event_flags`, `badges`) this
depends on, per `game_state.py`, rather than the whole `GameState`, so this
module stays testable as pure data-in/data-out logic.

Event flag IDs are bit positions within `wEventFlags`, sourced from
`pret/pokered`'s `constants/event_constants.asm` and computed by walking its
`const_def`/`const`/`const_skip`/`const_next` enumeration macros
(`macros/const.asm`) by hand from that file - the same numbering
`game_state.py`'s `_read_event_flags` produces (`offset*8+bit`, relative to
`_EVENT_FLAGS_START_ADDRESS` == `wEventFlags`'s own start address), so a
`GameState.event_flags` snapshot can be passed straight into this module's
flag IDs without translation. Badge checks use the same badge-name strings
`game_state.py`'s `_read_badges` produces (`items.py`'s `BOULDERBADGE` etc),
so `GameState.badges` also passes straight through.

Map targets are `pret/pokered`'s `constants/map_constants.asm` IDs, resolved
to display names via `lookup.map_name` (see `lookup/maps.py`) - the same
source/derivation `game_state.py` uses for `GameState.map_name`.

Tile-level `target_x`/`target_y` for all 14 milestones are populated below,
read off `pokemon_red.gb` itself via `rom_maps.parse_map`'s object records
(#97) rather than hand-documented - see `milestone_targets.py` for the
milestone -> object-record selection (a deliberate, one-line-rationale-per-
entry judgment call for maps with more than one candidate object) and
`tests/test_rom_maps.py` for the re-derivation check that keeps the literals
below honest against the ROM. This module itself still takes no ROM/PyBoy
dependency at import or call time - the values are plain data, generated
once and pinned here, not resolved live on every `track_milestones` call.

Unlike `game_state.py`'s address table, the event flag bit positions above
were computed by hand from `constants/event_constants.asm`'s source and have
*not* been re-verified by booting `pokemon_red.gb` and checking the flag
actually flips at that bit - a follow-up worth doing before relying on this
module against a live PyBoy instance.
"""

from dataclasses import dataclass

from jev_plays_pokemon.lookup import map_name

# Map IDs, per `constants/map_constants.asm`'s `const_def` position.
_MAP_OAKS_LAB = 40
_MAP_VIRIDIAN_MART = 42
_MAP_VIRIDIAN_GYM = 45
_MAP_PEWTER_GYM = 54
_MAP_CERULEAN_GYM = 65
_MAP_VERMILION_GYM = 92
_MAP_CELADON_GYM = 134
_MAP_POKEMON_TOWER_7F = 148
_MAP_FUCHSIA_GYM = 157
_MAP_CINNABAR_GYM = 166
_MAP_SAFFRON_GYM = 178
_MAP_BILLS_HOUSE = 88
_MAP_CHAMPIONS_ROOM = 120

# Event flag bit positions, per `constants/event_constants.asm`'s
# `const_def`/`const_skip`/`const_next` enumeration (see module docstring).
_EVENT_GOT_STARTER = 34
_EVENT_GOT_OAKS_PARCEL = 57  # Viridian Mart clerk hands over the parcel.
_EVENT_GOT_POKEDEX = 37  # Set alongside EVENT_OAK_GOT_PARCEL on delivery.
_EVENT_GOT_POKE_FLUTE = 296  # Rescuing Mr. Fuji in Pokemon Tower 7F.
_EVENT_GOT_SS_TICKET = 1372  # Bill's House, after helping Bill.
_EVENT_BEAT_CHAMPION_RIVAL = 2305  # Final battle, Champion's Room.


@dataclass(frozen=True)
class MilestoneTarget:
    map_id: int
    map_name: str
    # Tile-level destination within `map_id`, in the same world coordinates
    # as `GameState.player_x`/`player_y` - populated for all 14 milestones
    # below (see module docstring). `None` remains a valid value the type
    # allows - it tells `navigation.py`'s `resolve_navigation_target` there's
    # nothing tile-precise to path toward - for any future milestone added
    # without a resolved target yet.
    target_x: int | None = None
    target_y: int | None = None


@dataclass(frozen=True)
class Milestone:
    milestone_id: str
    description: str
    target: MilestoneTarget


@dataclass(frozen=True)
class MilestoneProgress:
    completed: tuple[Milestone, ...]
    current: Milestone | None
    future: tuple[Milestone, ...]


def _target(map_id: int, target_x: int, target_y: int) -> MilestoneTarget:
    return MilestoneTarget(
        map_id=map_id,
        map_name=map_name(map_id),
        target_x=target_x,
        target_y=target_y,
    )


@dataclass(frozen=True)
class _MilestoneCheck:
    milestone: Milestone
    event_flag: int | None = None
    badge: str | None = None

    def __post_init__(self) -> None:
        if (self.event_flag is None) == (self.badge is None):
            raise ValueError(
                f"{self.milestone.milestone_id}: exactly one of event_flag/badge must be set"
            )

    def is_complete(self, event_flags: frozenset[int], badges: tuple[str, ...]) -> bool:
        if self.event_flag is not None:
            return self.event_flag in event_flags
        return self.badge in badges


_MILESTONES: tuple[_MilestoneCheck, ...] = (
    _MilestoneCheck(
        Milestone(
            "got_starter",
            "Choose a starter Pokemon from Professor Oak",
            _target(_MAP_OAKS_LAB, 8, 3),  # milestone_targets.py: "got_starter"
        ),
        event_flag=_EVENT_GOT_STARTER,
    ),
    _MilestoneCheck(
        Milestone(
            "got_oaks_parcel",
            "Pick up Oak's Parcel from the Viridian City Poke Mart",
            _target(
                _MAP_VIRIDIAN_MART, 3, 3
            ),  # milestone_targets.py: "got_oaks_parcel"
        ),
        event_flag=_EVENT_GOT_OAKS_PARCEL,
    ),
    _MilestoneCheck(
        Milestone(
            "got_pokedex",
            "Deliver Oak's Parcel and receive the Pokedex",
            _target(_MAP_OAKS_LAB, 5, 2),  # milestone_targets.py: "got_pokedex"
        ),
        event_flag=_EVENT_GOT_POKEDEX,
    ),
    _MilestoneCheck(
        Milestone(
            "boulder_badge",
            "Defeat Brock for the Boulder Badge",
            _target(_MAP_PEWTER_GYM, 4, 1),  # milestone_targets.py: "boulder_badge"
        ),
        badge="BOULDERBADGE",
    ),
    _MilestoneCheck(
        Milestone(
            "cascade_badge",
            "Defeat Misty for the Cascade Badge",
            _target(_MAP_CERULEAN_GYM, 4, 2),  # milestone_targets.py: "cascade_badge"
        ),
        badge="CASCADEBADGE",
    ),
    # Bill's House (Route 25, north of Cerulean) comes next chronologically:
    # the S.S. Ticket it rewards is required to board the S.S. Anne out of
    # Vermilion, which sits ahead of that city's Thunder Badge below.
    _MilestoneCheck(
        Milestone(
            "got_ss_ticket",
            "Help Bill and receive the S.S. Ticket",
            _target(_MAP_BILLS_HOUSE, 4, 4),  # milestone_targets.py: "got_ss_ticket"
        ),
        event_flag=_EVENT_GOT_SS_TICKET,
    ),
    _MilestoneCheck(
        Milestone(
            "thunder_badge",
            "Defeat Lt. Surge for the Thunder Badge",
            _target(_MAP_VERMILION_GYM, 5, 1),  # milestone_targets.py: "thunder_badge"
        ),
        badge="THUNDERBADGE",
    ),
    _MilestoneCheck(
        Milestone(
            "rainbow_badge",
            "Defeat Erika for the Rainbow Badge",
            _target(_MAP_CELADON_GYM, 4, 3),  # milestone_targets.py: "rainbow_badge"
        ),
        badge="RAINBOWBADGE",
    ),
    _MilestoneCheck(
        Milestone(
            "got_poke_flute",
            "Rescue Mr. Fuji in Pokemon Tower and receive the Poke Flute",
            _target(
                _MAP_POKEMON_TOWER_7F, 10, 3
            ),  # milestone_targets.py: "got_poke_flute"
        ),
        event_flag=_EVENT_GOT_POKE_FLUTE,
    ),
    _MilestoneCheck(
        Milestone(
            "soul_badge",
            "Defeat Koga for the Soul Badge",
            _target(_MAP_FUCHSIA_GYM, 4, 10),  # milestone_targets.py: "soul_badge"
        ),
        badge="SOULBADGE",
    ),
    _MilestoneCheck(
        Milestone(
            "marsh_badge",
            "Defeat Sabrina for the Marsh Badge",
            _target(_MAP_SAFFRON_GYM, 9, 8),  # milestone_targets.py: "marsh_badge"
        ),
        badge="MARSHBADGE",
    ),
    _MilestoneCheck(
        Milestone(
            "volcano_badge",
            "Defeat Blaine for the Volcano Badge",
            _target(_MAP_CINNABAR_GYM, 3, 3),  # milestone_targets.py: "volcano_badge"
        ),
        badge="VOLCANOBADGE",
    ),
    _MilestoneCheck(
        Milestone(
            "earth_badge",
            "Defeat Giovanni for the Earth Badge",
            _target(_MAP_VIRIDIAN_GYM, 2, 1),  # milestone_targets.py: "earth_badge"
        ),
        badge="EARTHBADGE",
    ),
    _MilestoneCheck(
        Milestone(
            "beat_champion",
            "Defeat the rival as Champion",
            _target(_MAP_CHAMPIONS_ROOM, 4, 2),  # milestone_targets.py: "beat_champion"
        ),
        event_flag=_EVENT_BEAT_CHAMPION_RIVAL,
    ),
)


def track_milestones(
    event_flags: frozenset[int], badges: tuple[str, ...]
) -> MilestoneProgress:
    """Split the scripted milestone list into completed / current / future.

    `current` is the first not-yet-completed milestone in list order - the
    navigation macro's destination - and is `None` once every milestone is
    complete.
    """
    completed: list[Milestone] = []
    current: Milestone | None = None
    future: list[Milestone] = []

    for check in _MILESTONES:
        if current is not None:
            future.append(check.milestone)
        elif check.is_complete(event_flags, badges):
            completed.append(check.milestone)
        else:
            current = check.milestone

    return MilestoneProgress(
        completed=tuple(completed), current=current, future=tuple(future)
    )
