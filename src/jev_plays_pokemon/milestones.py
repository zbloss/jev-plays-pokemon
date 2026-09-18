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

Targets are map-level only, not per-milestone tile coordinates: no source in
this repo documents a verified in-map destination tile for these beats (e.g.
where Brock stands in Pewter Gym), and guessing one wouldn't be verifiable
without booting the ROM. The navigation macro ticket can add coordinates
once it has a way to verify them, same as this module can be extended then.

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


def _target(map_id: int) -> MilestoneTarget:
    return MilestoneTarget(map_id=map_id, map_name=map_name(map_id))


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
            _target(_MAP_OAKS_LAB),
        ),
        event_flag=_EVENT_GOT_STARTER,
    ),
    _MilestoneCheck(
        Milestone(
            "got_oaks_parcel",
            "Pick up Oak's Parcel from the Viridian City Poke Mart",
            _target(_MAP_VIRIDIAN_MART),
        ),
        event_flag=_EVENT_GOT_OAKS_PARCEL,
    ),
    _MilestoneCheck(
        Milestone(
            "got_pokedex",
            "Deliver Oak's Parcel and receive the Pokedex",
            _target(_MAP_OAKS_LAB),
        ),
        event_flag=_EVENT_GOT_POKEDEX,
    ),
    _MilestoneCheck(
        Milestone(
            "boulder_badge",
            "Defeat Brock for the Boulder Badge",
            _target(_MAP_PEWTER_GYM),
        ),
        badge="BOULDERBADGE",
    ),
    _MilestoneCheck(
        Milestone(
            "cascade_badge",
            "Defeat Misty for the Cascade Badge",
            _target(_MAP_CERULEAN_GYM),
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
            _target(_MAP_BILLS_HOUSE),
        ),
        event_flag=_EVENT_GOT_SS_TICKET,
    ),
    _MilestoneCheck(
        Milestone(
            "thunder_badge",
            "Defeat Lt. Surge for the Thunder Badge",
            _target(_MAP_VERMILION_GYM),
        ),
        badge="THUNDERBADGE",
    ),
    _MilestoneCheck(
        Milestone(
            "rainbow_badge",
            "Defeat Erika for the Rainbow Badge",
            _target(_MAP_CELADON_GYM),
        ),
        badge="RAINBOWBADGE",
    ),
    _MilestoneCheck(
        Milestone(
            "got_poke_flute",
            "Rescue Mr. Fuji in Pokemon Tower and receive the Poke Flute",
            _target(_MAP_POKEMON_TOWER_7F),
        ),
        event_flag=_EVENT_GOT_POKE_FLUTE,
    ),
    _MilestoneCheck(
        Milestone(
            "soul_badge", "Defeat Koga for the Soul Badge", _target(_MAP_FUCHSIA_GYM)
        ),
        badge="SOULBADGE",
    ),
    _MilestoneCheck(
        Milestone(
            "marsh_badge",
            "Defeat Sabrina for the Marsh Badge",
            _target(_MAP_SAFFRON_GYM),
        ),
        badge="MARSHBADGE",
    ),
    _MilestoneCheck(
        Milestone(
            "volcano_badge",
            "Defeat Blaine for the Volcano Badge",
            _target(_MAP_CINNABAR_GYM),
        ),
        badge="VOLCANOBADGE",
    ),
    _MilestoneCheck(
        Milestone(
            "earth_badge",
            "Defeat Giovanni for the Earth Badge",
            _target(_MAP_VIRIDIAN_GYM),
        ),
        badge="EARTHBADGE",
    ),
    _MilestoneCheck(
        Milestone(
            "beat_champion",
            "Defeat the rival as Champion",
            _target(_MAP_CHAMPIONS_ROOM),
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
