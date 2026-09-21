"""Action-execution: turns a chosen tactical action into emulator input.

Two action kinds, per `CONTEXT.md`'s "Navigation macro" entry and the parent
spec's (#14) action-space decision:

- a raw Game Boy button press, executed directly against PyBoy;
- the navigation macro, which takes no destination argument - it reads its
  target from the current-objective milestone tracker (`milestones.py`,
  #18) - and walks the player toward it via an internally-implemented A*
  pathfind over PyBoy's RAM/VRAM-derived local collision map, executed over
  multiple emulator frames.

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

Local, not global, pathfinding: PyBoy 2.2.0's Gen1 wrapper only exposes a
collision map for the visible on-screen window (`game_area_collision`,
scrolled to follow the player) - Game Boy VRAM never holds a whole map's
tile data at once, and this pinned PyBoy version doesn't expose the
WRAM-buffered full-map block data either (see `game_state.py`'s docstring
for the same 2.2.0-vs-dev-branch API gap). The macro therefore re-plans a
fresh local path every step from whatever's on screen right then, rather
than computing one global route up front; it walks *toward* the target, it
doesn't guarantee arrival.

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
same-map-only behavior, rather than a crash. Local A* also treats every
other known hop tile on the player's current map as an obstacle by default
(`_walkable`) - it won't route the player onto a warp/connection tile that
isn't the one it's actually trying to reach - except the current step's own
intended goal tile, which is always walkable (ADR-0002's "context-dependent"
warp treatment).

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
import math
from dataclasses import dataclass
from functools import lru_cache

from pyboy import PyBoy

from jev_plays_pokemon import rom_maps
from jev_plays_pokemon.game_state import extract_game_state
from jev_plays_pokemon.milestones import Milestone
from jev_plays_pokemon.travel_graph import (
    MILESTONE_MAP_IDS,
    TravelGraph,
    build_milestone_travel_graph,
    next_hop,
)

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

# `game_area_collision()` returns an (18, 20) grid, doubled up from a 9x10
# block grid so each 2x2 tile block shares one value (see the Gen1 wrapper's
# `_get_screen_walkable_matrix`/`game_area_collision`). The player's own
# tile always lands on the block at raw-tile (row=8, col=8) - verified in
# tests/test_navigation.py by checking that cell against the real, visible
# doorway gap in a booted `pokemon_red.gb`'s Pallet Town collision map.
_PLAYER_GRID_COL = 8
_PLAYER_GRID_ROW = 8


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


def _walkable(
    collision,
    col: int,
    row: int,
    warp_cells: frozenset[tuple[int, int]],
    goal: tuple[int, int],
) -> bool:
    """Whether `(col, row)` is walkable: on the collision grid, not blocked
    terrain, and - per ADR-0002's context-dependent warp treatment - not a
    known hop tile from `warp_cells` unless it's `goal`, this step's own
    intended destination (the next hop's tile, or the milestone's own final
    tile - see `execute_navigation_macro`)."""
    height, width = collision.shape
    if not (0 <= col < width and 0 <= row < height and collision[row, col] != 0):
        return False
    return (col, row) == goal or (col, row) not in warp_cells


def _next_step_toward(
    collision,
    goal_col: int,
    goal_row: int,
    warp_cells: frozenset[tuple[int, int]] = frozenset(),
) -> str | None:
    """A* from the player's fixed grid cell toward `(goal_col, goal_row)`.

    The goal is expressed in the same screen-relative grid as `collision`
    and may sit outside it, or on a tile it can't resolve as walkable, since
    the target is often further away than the locally visible window (see
    module docstring). When the exact goal can't be reached, this returns
    the first step of the shortest path toward whichever *reachable* tile
    ends up closest to it by Manhattan distance, so the macro still makes
    real progress instead of giving up. Returns `None` if the player's own
    cell has no walkable neighbours at all.

    `warp_cells` are this map's known hop tiles (screen-relative, see
    `execute_navigation_macro`), treated as obstacles by `_walkable` unless
    a cell is `(goal_col, goal_row)` itself - ADR-0002's context-dependent
    warp treatment.
    """
    start = (_PLAYER_GRID_COL, _PLAYER_GRID_ROW)
    goal = (goal_col, goal_row)

    def heuristic(cell: tuple[int, int]) -> int:
        return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])

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
            neighbor = (current[0] + dx, current[1] + dy)
            if neighbor in closed or not _walkable(
                collision, *neighbor, warp_cells, goal
            ):
                continue
            tentative_cost = cost + 1
            if tentative_cost < best_cost.get(neighbor, math.inf):
                best_cost[neighbor] = tentative_cost
                came_from[neighbor] = current
                heapq.heappush(
                    open_heap,
                    (tentative_cost + heuristic(neighbor), tentative_cost, neighbor),
                )

    if best_cell == start:
        return None

    node = best_cell
    while came_from[node] != start:
        node = came_from[node]
    step = (node[0] - start[0], node[1] - start[1])
    return _DIRECTION_BY_DELTA.get(step)


def _screen_cell(
    player_x: int, player_y: int, world_x: int, world_y: int
) -> tuple[int, int]:
    """`(world_x, world_y)` as a screen-relative grid cell, anchored off the
    player's own current world position - the same world-to-screen delta
    both a step's goal tile and a known hop tile need converting through."""
    return (
        _PLAYER_GRID_COL + world_x - player_x,
        _PLAYER_GRID_ROW + world_y - player_y,
    )


def _warp_cells_on_screen(
    map_id: int, player_x: int, player_y: int
) -> frozenset[tuple[int, int]]:
    """Screen-relative grid cells for every hop tile known to leave
    `map_id` (`travel_graph.py`'s `hops_from`), anchored off the player's
    current world position. Empty when the graph has no hops for this map
    yet - degrading to plain collision-only pathing."""
    return frozenset(
        _screen_cell(player_x, player_y, hop.from_x, hop.from_y)
        for hop in _travel_graph().hops_from(map_id)
    )


def execute_navigation_macro(
    pyboy: PyBoy, target: NavigationTarget, max_steps: int = 128
) -> bool:
    """Walk the player toward `target` over multiple emulator frames.

    Re-reads position and re-plans from the freshly-read local collision
    grid before every step (see module docstring), rather than committing
    to one upfront route. Returns whether it moved the player at all.

    Cross-map routing (ADR-0002, #102): on every step, if the player isn't
    on `target.map_id` yet, this queries `travel_graph.py`'s hop graph for
    the next hop from wherever the player currently is toward
    `target.map_id`, and paths local A* toward that hop's tile on the
    current map instead of `target`'s own tile - walking onto it triggers
    the game's own map transition, and the following step (in this call or
    a later one) picks up the next hop from the new map. If the graph has
    no route from the current map yet (ADR-0002's incremental build), this
    is a no-op for the milestone, matching pre-#102 same-map-only behavior,
    rather than a crash. Once on `target.map_id`, this paths straight to
    `target`'s own tile, unchanged from before #102.

    `max_steps` (default 128, up from an earlier 32) bounds how far a
    single call walks before returning - the local A* below only ever gives
    up when the player's own cell has no walkable neighbour at all, so in
    practice this cap, not a real decision point, is what ends most calls.
    128 is meant to cross a typical multi-screen corridor in one Jev turn
    (cutting down on redundant "keep going to the same place" calls) while
    still being bounded against a maze-like dead-end pocket that this
    local-only planner has no memory of previously-visited cells to avoid.

    Also returns early - before touching the collision grid or pressing
    anything - the instant `state.dialog_open` or `state.battle.in_battle`
    is true, even mid-walk (e.g. a sight-triggered trainer). That keeps a
    long `max_steps` budget from turning into a burst of blind presses
    against a battle menu or a dialog box: control goes back to `run_turn`
    so Jev gets a fresh, per-turn decision there, exactly as it already does
    for every other in-battle/dialog action.
    """
    moved = False
    for _ in range(max_steps):
        state = extract_game_state(pyboy)
        if state.dialog_open or state.battle.in_battle:
            break

        if state.map_id == target.map_id:
            goal_x, goal_y = target.x, target.y
        else:
            hop = next_hop(_travel_graph(), state.map_id, target.map_id)
            if hop is None:
                break
            goal_x, goal_y = hop.from_x, hop.from_y

        if goal_x == state.player_x and goal_y == state.player_y:
            break

        collision = pyboy.game_area_collision()
        goal_col, goal_row = _screen_cell(
            state.player_x, state.player_y, goal_x, goal_y
        )
        warp_cells = _warp_cells_on_screen(state.map_id, state.player_x, state.player_y)
        step = _next_step_toward(collision, goal_col, goal_row, warp_cells)
        if step is None:
            break

        execute_button(pyboy, step)
        moved = True

    return moved
