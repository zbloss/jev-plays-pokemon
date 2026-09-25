"""Action-execution: turns a chosen tactical action into emulator input.

Two action kinds, per `CONTEXT.md`'s "Navigation macro" entry and the parent
spec's (#14) action-space decision:

- a raw Game Boy button press, executed directly against PyBoy;
- the navigation macro, which takes no destination argument - it reads its
  target from the current-objective milestone tracker (`milestones.py`,
  #18) - and walks the player toward it via an internally-implemented A*
  pathfind over the map decoded from this repo's own ROM (see "Walkability
  comes from the ROM" below), executed over multiple emulator frames.

No code is reused from ClaudePlaysPokemonStarter (see #9's licensing
decision): both the button-hold timing below and the A* search are written
independently against PyBoy 2.2.0's public API, and verified by booting this
repo's own `pokemon_red.gb` (see `tests/test_navigation.py` - which also
documents the exact input sequence needed to get PyBoy past the
un-skippable name-entry intro into a controllable, real game state).

Button timing: a single directional press only turns the player to face
that way if they weren't already facing it - a second press (or a long
enough hold) is what actually walks them - and once moving, the player
keeps walking one tile every ~16 frames for as long as the button stays
held. PyBoy 2.2.0's own docs describe `PyBoy.button(button, delay)` as
holding for `delay` calls to `PyBoy.tick`, but that count is calls, not
frames: `pyboy.button('down', 24); pyboy.tick(24, False)` releases the
button after that single `tick` call, not 24 frames into it - so
`execute_button` drives `button_press`/`button_release` across
individually-ticked frames instead. A *fixed* hold length can't reliably
produce "exactly one tile": whether the first frame needs to spend time
turning varies, so a hold long enough to cover a turn will, when no turn
was needed, run long enough to walk two or three tiles instead of one (seen
by holding a fixed 48 frames against a real boot - see this ticket's own
verification). `execute_button` instead holds a direction only until the
player's own RAM position actually changes (capped at `_MAX_WALK_FRAMES` in
case the input is blocked and never moves), so it always advances by
whatever one input naturally does - one tile normally, more on a terrain
feature like a ledge that forces a longer hop - and never overshoots by
continuing to hold past that.

Walkability comes from the ROM, not from the emulator's screen: the A* searches a
`MapWalkability` that `_map_walkability` decodes out of `pokemon_red.gb` with
`tileset_collision.py` - the same source the verified milestone walks are planned
with - and plans in world tiles, the coordinates `GameState.player_x`/`player_y`
already report. PyBoy's `game_area_collision()` is deliberately not consulted. It
exposes only the visible window, and measured at three live Pallet Town positions
against the ROM's own passable lists it disagrees with them on 117-181 of its 360
cells (and calls 284 cells walkable where the ROM has 252), so a plan built from
it routes onto tiles the ROM then refuses - which is exactly how a walk ends up
pressing into one wall for its whole `max_steps` budget while reporting that it is
making progress. It is also blind to `CheckForTilePairCollisions`, the elevation
pairs that make two individually-walkable tiles impossible to walk between
(`tileset_collision.py`'s Mt Moon B2F measurement: 483 "walkable" tiles around
that stair landing, 67 of them legally reachable). Searching the whole map costs
nothing extra: it decodes from the ROM in one pass and is cached per map. The
macro still re-plans on every step, so it walks *toward* the target and does not
guarantee arrival - it just no longer plans against a map that isn't there.

Cross-map routing (`docs/adr/0002-travel-graph-for-cross-map-navigation.md`,
#102): when the player isn't on the current milestone's target map yet,
`execute_navigation_macro` queries `travel_graph.py`'s (#101) hand-authored
hop graph, on every step, for the next hop from wherever the player
currently is toward the milestone's target map, and hands local A* the
current map's side of that hop's tile instead of the milestone's own tile -
once local A* walks the player onto it, the game's own warp/connection
handling does the actual map transition, and the next step (possibly the
next `execute_navigation_macro` call entirely) picks up the following hop
from there. No route is planned or cached up front: each step re-derives
the next hop from current game state, per ADR-0002's "Statelessness"
section, so a player knocked off course (a battle, a stray screen, an NPC)
still makes progress from wherever they actually ended up on the very next
call. A milestone whose target map has no known route in the graph yet
(ADR-0002's incremental build) is a graceful no-op, matching today's
same-map-only behavior, rather than a crash. The A* also treats every
other known hop tile on the player's current map as an obstacle by default
(`_next_step_toward`'s `avoid` set, built by `_hop_tiles`) - it won't route the
player onto a warp/connection tile that isn't the one it's actually trying to
reach - except the current step's own intended goal tile, which is always
allowed (ADR-0002's "context-dependent" warp treatment).

All 14 scripted milestones now carry ROM-derived tile coordinates
(`milestones.py`, #98), so `resolve_navigation_target` resolves a real
`NavigationTarget` for any of them; it still returns `None` when there's no
current milestone (every milestone complete), when `target_x`/`target_y`
are unset - for some future milestone added without a resolved target, or
when the player isn't on the milestone's target map and the travel graph
doesn't (yet) know a route there. `execute_navigation_macro` is still
exercised directly with an explicit `NavigationTarget` in most tests (see
tests) for isolation from milestone-tracking state, not because real
milestones lack one.
"""

from __future__ import annotations

import heapq
import logging
import math
from dataclasses import dataclass
from functools import cache, lru_cache

from pyboy import PyBoy

from jev_plays_pokemon import rom_maps
from jev_plays_pokemon.game_state import extract_game_state
from jev_plays_pokemon.milestones import Milestone
from jev_plays_pokemon.tileset_collision import (
    TilesetHeader,
    crosses_blocked_pair,
    is_walkable,
    parse_tile_pair_collisions,
    parse_tileset_headers,
)
from jev_plays_pokemon.travel_graph import (
    MILESTONE_MAP_IDS,
    TravelGraph,
    build_milestone_travel_graph,
    next_hop,
)

logger = logging.getLogger(__name__)

RAW_BUTTONS: tuple[str, ...] = (
    "up",
    "down",
    "left",
    "right",
    "a",
    "b",
    "start",
    "select",
)

# (delta-x, delta-y) in world tile coordinates - matches `GameState.player_x`/
# `player_y`'s axes (x grows right, y grows down), confirmed against a real
# boot in tests/test_navigation.py.
_DIRECTIONS: dict[str, tuple[int, int]] = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}
_DIRECTION_BY_DELTA: dict[tuple[int, int], str] = {
    delta: direction for direction, delta in _DIRECTIONS.items()
}

# See module docstring's "Button timing" section.
_MAX_WALK_FRAMES = 64
_PRESS_FRAMES = 8
_SETTLE_FRAMES = 16

# `wXCoord`/`wYCoord` - the same addresses `game_state.py` reads into
# `GameState.player_x`/`player_y` (see that module's docstring for their
# derivation/verification). Read directly here, rather than through
# `extract_game_state`, since `execute_button` polls them once per frame
# while a direction is held and doesn't need the rest of `GameState`.
_PLAYER_X_ADDRESS = 0xD362
_PLAYER_Y_ADDRESS = 0xD361

# Frames to let a map transition finish before planning the step after it.
# `execute_button` releases as soon as the player's RAM position changes, and a
# warp/stair transition changes it immediately - well before the new map's
# tiles are loaded and the fade ends. Planning from that mid-transition read
# would press a direction the ROM then acts on *after* the transition lands,
# which can shove the player back onto the warp tile they just came through.
# Measured against a real boot: the position `execute_navigation_macro` reads 16
# settle frames after crossing Red's House's front door is not the position the
# player ends up at (it reads Pallet Town (3, 7) mid-transition and settles to
# (5, 6)), and 60 frames is comfortably past the transition.
_WARP_SETTLE_FRAMES = 60


def execute_button(pyboy: PyBoy, button: str) -> None:
    """Press `button` against PyBoy and let its in-game effect play out.

    For the four directions, holds the press only until the player's own
    RAM position changes (or `_MAX_WALK_FRAMES` passes, if the input is
    blocked and never moves) - see the module docstring for why a fixed
    hold length can't reliably do this. The two face buttons and start/
    select don't have a position to poll, so those just get a short press.
    """
    if button not in RAW_BUTTONS:
        raise ValueError(f"unknown button: {button!r}")

    pyboy.button_press(button)
    if button in _DIRECTIONS:
        memory = pyboy.memory
        before = (memory[_PLAYER_X_ADDRESS], memory[_PLAYER_Y_ADDRESS])
        for _ in range(_MAX_WALK_FRAMES):
            pyboy.tick(1, True)
            if (memory[_PLAYER_X_ADDRESS], memory[_PLAYER_Y_ADDRESS]) != before:
                break
    else:
        for _ in range(_PRESS_FRAMES):
            pyboy.tick(1, True)
    pyboy.button_release(button)
    for _ in range(_SETTLE_FRAMES):
        pyboy.tick(1, True)


# Gen 1's shared "current cursor position in whichever menu is on screen"
# register (`wCurrentMenuItem`, pret/pokered's `ram/wram.asm`) - reused across
# every battle-menu screen (the main FIGHT/PKMN/ITEM/RUN menu, the move list,
# the bag list, the party switch list) rather than one address per menu type,
# per that symbol's own doc comment in `wram.asm` ("the id of the currently
# selected menu item ... the top item has id 0"). Byte offset computed the
# same way `game_state.py`'s `_BATTLE_RESULT_ADDRESS` was: summing `wram.asm`'s
# `db` declarations forward from this run's first field, `wTopMenuItemY`,
# based at 0xCC24 per Data Crystal's RAM map - and cross-checked against two
# independent PyBoy-driven Pokemon Red battle bots reading this exact address
# live during real battles (`InsaneJSK/DeepRed`'s `battle_controller.py`;
# `2389-research/jev-plays-pokemon`'s `battle.py`, whose reading is validated
# against a captured real-battle save state, not address-table-only).
_MENU_CURSOR_ADDRESS = 0xCC26


def read_menu_cursor(pyboy: PyBoy) -> int:
    """Read Gen 1's live menu-cursor index for whatever battle-menu screen is
    currently on screen (`wCurrentMenuItem`) - see `_MENU_CURSOR_ADDRESS`."""
    return pyboy.memory[_MENU_CURSOR_ADDRESS]


def menu_list_delta_buttons(current: int, target: int) -> tuple[str, ...]:
    """The button sequence to move a vertical list menu's cursor from
    `current` to `target` - the delta only, never a fixed-length sequence
    (#76): Gen 1 remembers each list menu's cursor position across turns
    (pret/pokered's `wPartyAndBillsPCSavedMenuItem`/`wBagSavedMenuItem`/
    `wBattleAndStartSavedMenuItem`), so a hardcoded press count is only
    correct the one turn the cursor happens to already sit where it assumes -
    cross-repo research on `milanboers/jev-plays-pokemon` and
    `valentynkit/jev-plays-pokemon-red` both independently hit this exact
    failure mode (see #76).
    """
    delta = target - current
    button = "down" if delta >= 0 else "up"
    return (button,) * abs(delta)


def main_battle_menu_delta_buttons(current: int, target: int) -> tuple[str, ...]:
    """The button sequence to move the main FIGHT(0)/PKMN(1)/ITEM(2)/RUN(3)
    battle menu's cursor from `current` to `target`.

    This menu is a 2x2 grid, not a vertical list (row = index // 2, column =
    index % 2; DOWN/UP toggle the row, RIGHT/LEFT toggle the column) - so at
    most one press per axis gets there directly from wherever the cursor
    currently sits, with no wraparound assumption needed either way (#76).
    """
    current_row, current_col = divmod(current, 2)
    target_row, target_col = divmod(target, 2)
    buttons: list[str] = []
    if target_col != current_col:
        buttons.append("right" if target_col > current_col else "left")
    if target_row != current_row:
        buttons.append("down" if target_row > current_row else "up")
    return tuple(buttons)


@lru_cache(maxsize=1)
def _travel_graph() -> TravelGraph:
    """The travel graph (`travel_graph.py`, #101), built once from the
    pinned ROM and cached for the process lifetime.

    This caches fixed, ROM-derived hop *data*, not route state - ADR-0002's
    "Statelessness" section is about not persisting a chosen *route* across
    turns (`find_route`/`next_hop` still re-search from scratch on every
    call, per that module's own docstring); the graph itself never changes
    for a given ROM, so rebuilding it from scratch on every step/turn would
    just be wasted parsing.

    Falls back to an empty graph when `pokemon_red.gb` isn't present (it's
    gitignored - see `tests/test_navigation.py`'s own `ROM_PATH.exists()`
    gate): an empty graph makes every cross-map lookup resolve to "no known
    route", the same graceful-no-op signal ADR-0002 already defines for a
    milestone outside the graph's incrementally-built scope, rather than a
    crash. `decision.py`'s `resolve_navigation_target`/`out_of_battle_
    action_space` call this on every turn regardless of whether the current
    milestone is same-map, so tests exercising those against a plain fake
    `GameState` - no PyBoy, no ROM, by design (see `decision.py`'s module
    docstring) - must not be forced to boot a real ROM just to determine an
    unrelated milestone's map is out of reach.
    """
    try:
        rom = rom_maps.load_rom()
    except FileNotFoundError:
        return TravelGraph()
    rom_maps_by_id = {
        map_id: rom_maps.parse_map(rom, map_id) for map_id in MILESTONE_MAP_IDS
    }
    return build_milestone_travel_graph(rom_maps_by_id)


@dataclass(frozen=True)
class NavigationTarget:
    map_id: int
    x: int
    y: int


def resolve_navigation_target(
    milestone: Milestone | None, current_map: int
) -> NavigationTarget | None:
    """Pull the navigation macro's destination from the current milestone.

    `current_map` is the player's current map ID: when it differs from the
    milestone's own target map, this checks that `travel_graph.py`'s (#101)
    hop graph actually knows a route there yet - if it doesn't (ADR-0002's
    incremental build), this returns `None` so the macro isn't offered/run
    for a milestone it can't make any progress toward, exactly as it
    already does for a milestone with no verified tile coordinates at all.
    The returned target is always the milestone's own tile, regardless of
    `current_map` - `execute_navigation_macro` resolves the next hop's tile
    itself, fresh, on every step it takes (see module docstring).

    Returns `None` when there's no current milestone (every milestone
    complete), when it has no verified tile-level target yet, or when
    `current_map` isn't the target map and no route there is known yet (see
    this module's docstring and `milestones.py`'s own).
    """
    if milestone is None:
        return None
    target = milestone.target
    if target.target_x is None or target.target_y is None:
        return None
    if current_map != target.map_id and (
        next_hop(_travel_graph(), current_map, target.map_id) is None
    ):
        return None
    return NavigationTarget(target.map_id, target.target_x, target.target_y)


# ---------------------------------------------------------------------------
# What's walkable, according to the ROM
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _rom_bytes() -> bytes | None:
    """This repo's pinned ROM bytes, or `None` when `pokemon_red.gb` isn't present.

    Every walkability question below is answered from the ROM, and it's
    gitignored (see `tests/test_navigation.py`'s own `ROM_PATH.exists()` gate),
    so its absence has to degrade rather than raise: `_map_walkability` returns
    `None`, `execute_navigation_macro` declines to walk, and the turn costs a
    no-op instead of a crash - the same graceful fallback `_travel_graph` uses
    for a missing ROM.
    """
    try:
        return rom_maps.load_rom()
    except FileNotFoundError:
        return None


@lru_cache(maxsize=1)
def _tileset_headers() -> tuple[TilesetHeader, ...] | None:
    """The ROM's tileset header table (each tileset's passable tile list), once.

    `ValueError` - what `parse_tileset_headers` raises when the bytes it anchors
    on aren't structurally the table it expects - is treated like a missing ROM:
    a ROM this module can't decode is a ROM it can't plan against, so the live
    loop degrades instead of dying. Logged, because `lru_cache` means this body
    runs at most once per process and a silently blind planner is exactly the
    failure mode this module keeps having to relearn.
    """
    rom = _rom_bytes()
    if rom is None:
        return None
    try:
        return parse_tileset_headers(rom)
    except ValueError:
        logger.warning("could not decode the ROM's tileset headers", exc_info=True)
        return None


@lru_cache(maxsize=1)
def _tile_pair_collisions() -> frozenset[tuple[int, int, int]] | None:
    """The ROM's blocked tile-pair crossings (`tileset_collision.py`), once."""
    rom = _rom_bytes()
    if rom is None:
        return None
    try:
        return parse_tile_pair_collisions(rom)
    except ValueError:
        logger.warning("could not decode the ROM's tile-pair collisions", exc_info=True)
        return None


@dataclass(frozen=True)
class MapWalkability:
    """One map's walkability, decoded once from the ROM and cached.

    Two facts, because Gen 1 consults both before it lets an ordinary step
    happen (`tileset_collision.py`'s docstring quotes the `home/overworld.asm`
    call sites): `walkable` is `CheckTilePassable`'s answer per tile, and
    `blocked_crossings` is `CheckForTilePairCollisions`' answer per *edge* - the
    elevation pairs that make two individually-walkable tiles impossible to walk
    between, which is an edge property no per-tile set can express.

    Tiles are world coordinates, the same ones `GameState.player_x`/`player_y`
    report, so a caller never converts through a screen window.
    """

    map_id: int
    width: int
    height: int
    walkable: frozenset[tuple[int, int]]
    blocked_crossings: frozenset[tuple[tuple[int, int], tuple[int, int]]]

    def is_walkable(self, tile: tuple[int, int]) -> bool:
        return tile in self.walkable

    def can_cross(self, here: tuple[int, int], there: tuple[int, int]) -> bool:
        """Whether the ROM lets a walk cross from `here` into `there` (both
        already known walkable). Both directions are stored: `CheckForTilePair-
        Collisions` matches the standing tile against either half of a pair, so
        the rule is two-way."""
        return (here, there) not in self.blocked_crossings


@cache
def _map_walkability(map_id: int) -> MapWalkability | None:
    """`MapWalkability` for `map_id`, or `None` when it can't be decoded.

    `None` covers a missing or undecodable ROM and a map whose records don't
    resolve (`rom_maps.parse_map` raises `ValueError` on the `UNUSED_MAP_*`
    placeholder slots among others) - in every case the macro declines to walk
    rather than planning against a guess.

    Cached per map because `execute_navigation_macro` asks on every step and a
    map's answer never changes for a given ROM. Building it costs one pass over
    the map's tiles plus one pair check per adjacent walkable pair: tens of
    milliseconds on the largest maps in scope (Route 4's 90x18 grid), paid once.
    """
    rom, headers, pairs = _rom_bytes(), _tileset_headers(), _tile_pair_collisions()
    if rom is None or headers is None or pairs is None:
        return None
    try:
        rmap = rom_maps.parse_map(rom, map_id)
    except ValueError:
        logger.warning("could not decode map %d from the ROM", map_id, exc_info=True)
        return None

    width, height = rmap.width_blocks * 2, rmap.height_blocks * 2
    walkable = frozenset(
        (x, y)
        for x in range(width)
        for y in range(height)
        if is_walkable(rom, rmap, headers, x, y)
    )

    blocked: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for x, y in sorted(walkable):
        for neighbour in ((x + 1, y), (x, y + 1)):
            if neighbour not in walkable:
                continue
            if crosses_blocked_pair(rom, rmap, headers, pairs, (x, y), neighbour):
                blocked.add(((x, y), neighbour))
                blocked.add((neighbour, (x, y)))

    return MapWalkability(
        map_id=map_id,
        width=width,
        height=height,
        walkable=walkable,
        blocked_crossings=frozenset(blocked),
    )


def _hop_tiles(map_id: int) -> frozenset[tuple[int, int]]:
    """Every hop tile leaving `map_id` (`travel_graph.py`'s `hops_from`), as
    world coordinates. Empty when the graph has no hops for this map yet."""
    return frozenset(
        (hop.from_x, hop.from_y) for hop in _travel_graph().hops_from(map_id)
    )


def _next_step_toward(
    walk: MapWalkability,
    start: tuple[int, int],
    goal: tuple[int, int],
    avoid: frozenset[tuple[int, int]] = frozenset(),
    refused: frozenset[tuple[tuple[int, int], tuple[int, int]]] = frozenset(),
) -> str | None:
    """The first direction to press to walk `start` toward `goal`, or `None`.

    A* over the map's own world tiles. `goal` may sit outside the map, or on a
    tile the ROM doesn't consider walkable at all (an item ball or an NPC is a
    legitimate milestone target you interact with from next door), so when the
    exact goal can't be reached this returns the first step toward whichever
    *reachable* tile ends up closest to it by Manhattan distance - the macro
    still makes real progress instead of giving up, as it always has. `None`
    when the player's own tile isn't walkable or has no legal step out of it.

    `avoid` holds this map's other hop tiles, treated as obstacles unless a tile
    is `goal` itself - ADR-0002's context-dependent warp treatment, so the
    player never gets routed onto a door that isn't the one this step aims at.

    `refused` holds the `(from, to)` edges the ROM has already declined during
    this walk (see `execute_navigation_macro`). It belongs here rather than only
    in the caller because this is the only place that decides which tiles are
    reachable at all: without it, a plan that steps into a tile the ROM refuses
    for a reason the passable list can't express - a ledge, which needs A to
    drop off, is the standing example - re-proposes that exact press for every
    remaining step and never gets anywhere.
    """
    if not walk.is_walkable(start):
        return None

    def heuristic(cell: tuple[int, int]) -> int:
        return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])

    def step_is_legal(here: tuple[int, int], there: tuple[int, int]) -> bool:
        if (here, there) in refused or not walk.is_walkable(there):
            return False
        if not walk.can_cross(here, there):
            return False
        return there == goal or there not in avoid

    open_heap: list[tuple[int, int, tuple[int, int]]] = [(heuristic(start), 0, start)]
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    best_cost: dict[tuple[int, int], int] = {start: 0}
    closed: set[tuple[int, int]] = set()
    best_cell = start
    best_heuristic = heuristic(start)

    while open_heap:
        _, cost, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)

        current_heuristic = heuristic(current)
        if current_heuristic < best_heuristic:
            best_cell, best_heuristic = current, current_heuristic
        if current == goal:
            best_cell = current
            break

        for dx, dy in _DIRECTIONS.values():
            neighbour = (current[0] + dx, current[1] + dy)
            if neighbour in closed or not step_is_legal(current, neighbour):
                continue
            tentative_cost = cost + 1
            if tentative_cost < best_cost.get(neighbour, math.inf):
                best_cost[neighbour] = tentative_cost
                came_from[neighbour] = current
                heapq.heappush(
                    open_heap,
                    (tentative_cost + heuristic(neighbour), tentative_cost, neighbour),
                )

    if best_cell == start:
        return None

    node = best_cell
    while came_from[node] != start:
        node = came_from[node]
    step = (node[0] - start[0], node[1] - start[1])
    return _DIRECTION_BY_DELTA.get(step)


def execute_navigation_macro(
    pyboy: PyBoy, target: NavigationTarget, max_steps: int = 128
) -> bool:
    """Walk the player toward `target` over multiple emulator frames.

    Re-reads the game state and re-plans from the ROM's own walkability for
    whatever map the player is actually on before every step (see module
    docstring), rather than committing to one upfront route. Returns whether the
    player's position changed at any point during the walk - `False` means the
    macro pressed nothing, or pressed and got nowhere.

    Cross-map routing (ADR-0002, #102): on every step, if the player isn't on
    `target.map_id` yet, this queries `travel_graph.py`'s hop graph for the next
    hop from wherever the player currently is toward `target.map_id`, and paths
    A* toward that hop's tile on the current map instead of `target`'s own tile -
    walking onto it triggers the game's own map transition, and the following
    step (in this call or a later one) picks up the next hop from the new map. If
    the graph has no route from the current map yet (ADR-0002's incremental
    build), or this map's tiles can't be decoded from the ROM
    (`_map_walkability` returning `None`), this is a no-op for the milestone
    rather than a crash. Once on `target.map_id`, this paths straight to
    `target`'s own tile.

    A step the ROM refuses is remembered for the rest of this walk and never
    re-proposed. `execute_button` holds a direction until the player's position
    changes, so a refused step costs a full `_MAX_WALK_FRAMES` and changes
    nothing - and because the A* below is stateless, re-planning from the same
    tile with the same map would name that same step again on every one of the
    `max_steps` iterations. That is how a walk can spend its whole budget
    pressing into one tile and still report that it moved: the press happens,
    the tile just was never going to accept it (a ledge, which the ROM only lets
    you step off with A held, is the case actually observed on Pallet Town's
    plateau edge). Refusing the edge is what turns that into a detour.

    A step that changes `map_id` is followed by `_WARP_SETTLE_FRAMES` of ticking
    before anything is planned from the new map - see that constant.

    `max_steps` (default 128, up from an earlier 32) bounds how far a single
    call walks before returning. 128 is meant to cross a typical multi-screen
    corridor in one Jev turn (cutting down on redundant "keep going to the same
    place" calls) while still being bounded against a maze-like dead-end pocket
    that this planner has no memory of previously-visited cells to avoid.

    Also returns early - before planning or pressing anything - the instant
    `state.dialog_open` or `state.battle.in_battle` is true, even mid-walk (e.g.
    a sight-triggered trainer). That keeps a long `max_steps` budget from turning
    into a burst of blind presses against a battle menu or a dialog box: control
    goes back to `run_turn` so Jev gets a fresh, per-turn decision there, exactly
    as it already does for every other in-battle/dialog action.
    """
    moved = False
    # The press whose outcome the next state read will settle: the position it
    # left the player at, and the tile it was aimed at. `None` once accounted
    # for. Settling one read late is deliberate - the position that says whether
    # a press worked is the one the next iteration reads anyway.
    pending: tuple[tuple[int, int, int], tuple[int, int]] | None = None
    last_position: tuple[int, int, int] | None = None
    refused: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    for _ in range(max_steps):
        state = extract_game_state(pyboy)
        position = (state.map_id, state.player_x, state.player_y)

        if pending is not None:
            if position == pending[0]:
                refused.add((pending[0][1:], pending[1]))
            else:
                moved = True
            pending = None

        if last_position is not None and position[0] != last_position[0]:
            for _ in range(_WARP_SETTLE_FRAMES):
                pyboy.tick(1, True)
            state = extract_game_state(pyboy)
            position = (state.map_id, state.player_x, state.player_y)
        last_position = position

        if state.dialog_open or state.battle.in_battle:
            break

        if state.map_id == target.map_id:
            goal = (target.x, target.y)
        else:
            hop = next_hop(_travel_graph(), state.map_id, target.map_id)
            if hop is None:
                break
            goal = (hop.from_x, hop.from_y)

        here = (state.player_x, state.player_y)
        if goal == here:
            break

        walk = _map_walkability(state.map_id)
        if walk is None:
            break

        step = _next_step_toward(
            walk, here, goal, _hop_tiles(state.map_id) - {goal}, frozenset(refused)
        )
        if step is None:
            break

        delta_x, delta_y = _DIRECTIONS[step]
        execute_button(pyboy, step)
        pending = (position, (here[0] + delta_x, here[1] + delta_y))

    if pending is not None:
        # The loop ended on an unsettled press; read once more so the return
        # value describes the whole walk rather than everything but its last
        # step.
        state = extract_game_state(pyboy)
        if (state.map_id, state.player_x, state.player_y) != pending[0]:
            moved = True

    return moved
