# Research: `pret/pokered` warp/map-connection data

Resolves issue [#87](https://github.com/zbloss/jev-plays-pokemon/issues/87), child of the wayfinder-map umbrella issue [#85](https://github.com/zbloss/jev-plays-pokemon/issues/85), itself following on from the cross-map architecture proposal in [#84](https://github.com/zbloss/jev-plays-pokemon/issues/84).

## Question

What warp/map-connection data does the `pret/pokered` disassembly expose per map —
door/stair/ledge/exit locations, destination map + destination tile — and in what
coordinate frame relative to `constants/map_constants.asm`'s map IDs?

## Top-line answer

`pret/pokered` exposes exactly the data #84's proposed waypoint+warp graph would
need, in a directly-usable form:

- **Where it lives:** one `data/maps/objects/<MapName>.asm` file per map, each with
  a `def_warp_events` / `warp_event` table (door/stair/most exits) built by the
  `warp_event` macro (`macros/scripts/maps.asm`). Map-edge walk-offs (routes bleeding
  into each other, not "warps" in the game's own vocabulary) are a *separate*
  mechanism: `connection` entries in each map's `data/maps/headers/<MapName>.asm`
  header. Ledges and cuttable trees are neither — they're per-tileset collision
  overrides with no map/destination data at all.
- **Coordinate frame:** `warp_event`'s X/Y and `connection`'s alignment values are
  in the exact same units as `wXCoord`/`wYCoord` — the same two RAM addresses this
  repo's `game_state.py` already reads into `GameState.player_x`/`player_y`
  (`0xD362`/`0xD361`). This is proven directly by the game's own warp-detection code
  (`home/overworld.asm`'s `CheckWarpsNoCollision`/`CheckWarpsCollision`,
  `engine/overworld/player_state.asm`'s `IsPlayerStandingOnWarp`), which compares
  `wYCoord`/`wXCoord` byte-for-byte against the warp table's stored Y/X with **no
  shift, doubling, or offset** anywhere in the comparison. A `warp_event`'s X/Y (or
  a milestone's future `target_x`/`target_y`) can be used as-is against this repo's
  `player_x`/`player_y` — no translation needed.
- **Completeness:** all 14 of this repo's current milestone maps (`milestones.py`)
  have a populated `warp_event` table in `pret/pokered` — 100% coverage, verified
  file-by-file below. Champion's Room and Saffron Gym show the format handles the
  harder cases too (multi-destination exits, same-map teleport-puzzle "warps").

## Sources consulted (primary, all from `pret/pokered` at `master`, fetched via
`raw.githubusercontent.com`/`gh api` during this research pass)

1. `macros/scripts/maps.asm` — `warp_event`, `def_warp_events`, `def_warps_to`,
   `bg_event`, `connection`, `map_header` macro definitions:
   https://github.com/pret/pokered/blob/master/macros/scripts/maps.asm
2. `data/maps/objects/*.asm` — per-map warp/object/sign event tables (14 files
   fetched directly, see completeness table below), e.g.:
   https://github.com/pret/pokered/blob/master/data/maps/objects/OaksLab.asm
3. `data/maps/headers/*.asm` — per-map header + `connection` entries, e.g.:
   https://github.com/pret/pokered/blob/master/data/maps/headers/OaksLab.asm
4. `home/overworld.asm` — the actual runtime warp-detection/warp-execution code
   (`CheckWarpsNoCollision`, `CheckWarpsCollision`, `WarpFound1`/`WarpFound2`,
   `CheckMapConnections`, `LoadMapHeader`, `LoadDestinationWarpPosition`):
   https://github.com/pret/pokered/blob/master/home/overworld.asm
5. `engine/overworld/player_state.asm` — `IsPlayerStandingOnWarp`,
   `IsWarpTileInFrontOfPlayer`, `IsPlayerStandingOnDoorTileOrWarpTile`:
   https://github.com/pret/pokered/blob/master/engine/overworld/player_state.asm
6. `engine/overworld/ledges.asm` — `HandleLedges`, confirming ledges are tileset
   collision data, not warp/object data:
   https://github.com/pret/pokered/blob/master/engine/overworld/ledges.asm
7. `engine/overworld/tilesets.asm` — `LoadTilesetHeader`'s call into
   `LoadDestinationWarpPosition`:
   https://github.com/pret/pokered/blob/master/engine/overworld/tilesets.asm
8. `ram/wram.asm` — `wWarpEntries`, `wNumberOfWarps`, `wDestinationWarpID`,
   `wCurrentTileBlockMapViewPointer`/`wYCoord`/`wXCoord`/`wYBlockCoord`/
   `wXBlockCoord` declaration order:
   https://github.com/pret/pokered/blob/master/ram/wram.asm
9. `constants/map_data_constants.asm` — `LAST_MAP`, `MAX_WARP_EVENTS`,
   `MAX_BG_EVENTS`, `MAX_OBJECT_EVENTS`:
   https://github.com/pret/pokered/blob/master/constants/map_data_constants.asm
10. `constants/map_constants.asm` — map ID `const_def` enumeration, cross-checked
    against this repo's `milestones.py` map IDs:
    https://github.com/pret/pokered/blob/master/constants/map_constants.asm

## This repo's existing conventions (read first, per this ticket's instructions)

- `src/jev_plays_pokemon/milestones.py`'s docstring: map targets are
  `pret/pokered`'s `constants/map_constants.asm` IDs, "resolved to display names
  via `lookup.map_name`" — the module hand-walks that file's `const_def` enumeration
  macros (`macros/const.asm`) to get each numeric map ID. This research
  independently re-fetched `constants/map_constants.asm` and its inline hex
  comments (e.g. `map_const OAKS_LAB, 5, 6 ; $28`) and confirmed all 13 non-Oak's-Lab
  IDs used in `milestones.py` match exactly (`$28`=40, `$2A`=42, `$2D`=45, `$36`=54,
  `$41`=65, `$58`=88, `$5C`=92, `$78`=120, `$86`=134, `$8E..$94`=142..148 for the
  Pokemon Tower floors — `$94`=148 for 7F, `$9D`=157, `$A6`=166, `$B2`=178).
- `src/jev_plays_pokemon/navigation.py`'s `NavigationTarget`/`MilestoneTarget`:
  tile-level targets are "in the same world tile coordinates as
  `GameState.player_x`/`player_y` — x grows right, y grows down", read from
  `_PLAYER_X_ADDRESS = 0xD362` / `_PLAYER_Y_ADDRESS = 0xD361` — i.e. `wXCoord`/
  `wYCoord` directly, confirmed in `tests/test_navigation.py` against a real ROM
  boot. This is the frame this research checks `warp_event` data against below.
- `navigation.py`'s docstring already flags cross-map travel as unimplemented
  ("doesn't attempt cross-map routing ... if the player isn't already on the
  target's map, it's a no-op") — the gap issue #84/#85/#87 are working on.

## 1. Where warp data lives, and its exact format

### File layout

Each map has **two** separate `.asm` files in `pret/pokered`:

- `data/maps/headers/<MapName>.asm` — the `map_header` macro invocation (map ID,
  tileset, dimensions) plus any `connection` entries (map-edge walk-offs — see
  §1c).
- `data/maps/objects/<MapName>.asm` — the `<MapName>_Object` label: border-block
  byte, then three tables in fixed order: `def_warp_events`/`warp_event` (doors,
  stairs, most map-to-map exits), `def_bg_events`/`bg_event` (signs — text only, no
  destination), `def_object_events`/`object_event` (NPCs/items — no destination),
  ending with `def_warps_to <MAP_ID>` (see §1b).

Example, `data/maps/objects/OaksLab.asm`
(https://github.com/pret/pokered/blob/master/data/maps/objects/OaksLab.asm):

```
OaksLab_Object:
	db $3 ; border block

	def_warp_events
	warp_event  4, 11, LAST_MAP, 3
	warp_event  5, 11, LAST_MAP, 3

	def_bg_events
	...
	def_object_events
	...
	def_warps_to OAKS_LAB
```

### The `warp_event` macro

Defined in `macros/scripts/maps.asm`
(https://github.com/pret/pokered/blob/master/macros/scripts/maps.asm):

```
;\1 x position
;\2 y position
;\3 destination map (-1 = wLastMap)
;\4 destination warp id; starts at 1 (internally at 0)
MACRO warp_event
	db \2, \1, \4 - 1, \3
	...
ENDM
```

So each `warp_event` source line takes four arguments — **source X, source Y,
destination map ID (or `LAST_MAP`), destination warp index (1-based in source)**
— and assembles to **4 ROM bytes per warp, in `Y, X, warp_id-1, dest_map` order**
(note: byte order is Y-then-X, opposite of the source arguments, and the warp
index is stored 0-based). `ram/wram.asm`'s own declaration of the runtime copy of
this table confirms the same field order and count:

```
wWarpEntries:: ds MAX_WARP_EVENTS * 4 ; Y, X, warp ID, map ID
```

(https://github.com/pret/pokered/blob/master/ram/wram.asm, `MAX_WARP_EVENTS EQU 32`
per `constants/map_data_constants.asm`.)

`LAST_MAP` (`constants/map_constants.asm`: `DEF LAST_MAP EQU $ff`) is a sentinel destination
map meaning "wherever the player warped in from" (`wLastMap`) — used for the exit
side of building entrances (Oak's Lab's two doorway warps both go `LAST_MAP`, warp
index 3, i.e. back to whichever outdoor Pallet Town warp led in).

### `def_warps_to` and destination-index resolution

`def_warps_to <MAP_ID>` closes out a map's object file. Its only functional
purpose (beyond an `ASSERT` that the three event tables didn't overflow
`MAX_WARP_EVENTS`/`MAX_BG_EVENTS`/`MAX_OBJECT_EVENTS`) is to emit a
`warp_to`/`event_displacement` entry per warp for `wOverworldMap` bookkeeping — it
does **not** register anything into the destination map. That means **"destination
warp index" is purely positional**: warp index *N* (1-based in source, 0-based on
disk/in RAM) means "the *N*-th `warp_event` line, in file order, inside the
**destination** map's own `data/maps/objects/<DestMap>.asm`" — there is no
separate index/registry to look up; you read the destination file's own table and
count.

This is confirmed by the runtime code, not just inferred from the macro:
`home/overworld.asm`'s `LoadMapHeader` (~line 2081) copies exactly `wNumberOfWarps
* 4` bytes from the *currently-loading* map's own ROM object data straight into
`wWarpEntries` — so whichever map is `wCurMap` right now, `wWarpEntries` holds
*that* map's own warp table, indexed 0-based exactly as the disk format stores it.
`LoadDestinationWarpPosition` (`home/overworld.asm` ~line 2425) is the routine that
converts a "destination warp ID" into position data; its own doc comment states
plainly:

```
; function to load position data for destination warp when switching maps
; INPUT:
; a = ID of destination warp within destination map
LoadDestinationWarpPosition::
```

i.e. exactly "the Nth entry in the destination map's own warp table," matching the
macro comment ("destination warp id; starts at 1 (internally at 0)").

### `def_warps_to`'s counterpart: map connections (§1c)

Not every map-to-map transition is a `warp_event`. Contiguous overworld maps (e.g.
Route 1 into Pallet Town) instead use `connection` entries in the map's *header*
file (`data/maps/headers/<MapName>.asm`), via the `connection` macro
(`macros/scripts/maps.asm`). These describe a shared edge (north/south/west/east),
a block-offset alignment, and are resolved at runtime by
`home/overworld.asm`'s `CheckMapConnections` (only reached when no `warp_event`
matched) by walking `wCurMap`'s `wNorthConnectedMap`/`wSouthConnectedMap`/
`wWestConnectedMap`/`wEastConnectedMap` + `...Alignment` fields — set from the
header's `connection` data during `LoadMapHeader`. These aren't relevant to any of
the 14 milestone maps in this repo (all interiors/gyms, reached via doors, not
map edges) but are worth knowing about for #85's broader wayfinder-graph design,
since a route walked from one outdoor map to an adjacent one crosses via
`connection`, not `warp_event`.

### Ledges and cuttable trees: not warp data at all

`engine/overworld/ledges.asm`'s `HandleLedges` confirms ledges have **no
map/destination data whatsoever** — a per-tileset `LedgeTiles` table (keyed by
facing direction + the tile the player is standing on + the tile they're facing)
just forces an automatic 2-tile hop in place; there's no `warp_event`-style
X/Y/dest-map record for a ledge anywhere in the object data. Cuttable trees are
likewise pure tile-collision toggles (a `CUT`-flaggable tile becomes walkable),
not warp/object records. Anything building a graph from `warp_event`/`connection`
data alone will not see ledges or cut-trees — those stay in local A*'s domain
(consistent with `navigation.py`'s existing docstring, which already treats ledges
as "a terrain feature ... that forces a longer hop" handled by `execute_button`'s
position-polling, not pathfinding).

## 2. Coordinate frame vs. `wXCoord`/`wYCoord` (and this repo's `player_x`/`player_y`)

**Warp coordinates are in the exact same frame as `wXCoord`/`wYCoord` — full tile
coordinates, no doubling, no block/pixel conversion.** This is proven directly by
the runtime warp-matching code, not inferred:

`home/overworld.asm`'s `CheckWarpsNoCollision` (~line 391):

```
CheckWarpsNoCollision::
	...
	ld a, [wYCoord]
	ld d, a
	ld a, [wXCoord]
	ld e, a
	ld hl, wWarpEntries
CheckWarpsNoCollisionLoop::
	ld a, [hli] ; check if the warp's Y position matches
	cp d
	jr nz, CheckWarpsNoCollisionRetry1
	ld a, [hli] ; check if the warp's X position matches
	cp e
	jr nz, CheckWarpsNoCollisionRetry2
; if a match was found
	...
```

and `engine/overworld/player_state.asm`'s `IsPlayerStandingOnWarp`:

```
IsPlayerStandingOnWarp::
	...
	ld hl, wWarpEntries
.loop
	ld a, [wYCoord]
	cp [hl]
	jr nz, .nextWarp1
	inc hl
	ld a, [wXCoord]
	cp [hl]
	jr nz, .nextWarp2
	...
```

Both do a **direct byte-for-byte `cp`** between `wYCoord`/`wXCoord` and the stored
warp Y/X — no `sla`/`srl` (shift), no add/subtract, no lookup table in between.
Since `wXCoord` (`0xD362`) and `wYCoord` (`0xD361`) are the exact two addresses
this repo's `game_state.py` reads into `GameState.player_x`/`player_y`
(`navigation.py`'s own `_PLAYER_X_ADDRESS`/`_PLAYER_Y_ADDRESS` constants), a
`warp_event`'s X/Y taken straight from `pret/pokered` source (or extracted from a
compiled ROM's `wWarpEntries` at runtime) would compare equal to this repo's
`player_x`/`player_y` on the tile the warp sits on — **no translation needed**.

Two adjacent, narrower caveats worth flagging for whoever builds #84's graph:

- **`wYBlockCoord`/`wXBlockCoord` are a different, narrower field**, and easy to
  confuse with the warp table's Y/X. `engine/overworld/tilesets.asm`'s
  `LoadTilesetHeader` computes them as `wYCoord AND $1` / `wXCoord AND $1` (just
  the low bit of the *tile* coordinate) purely for picking which 2x2-block variant
  of dungeon tile data to draw — this is a rendering-alignment detail, not a
  coordinate system warps or the player-position fields use for comparison. Do
  not use it as a "block coordinate" conversion factor for warp data; none is
  needed.
  Confirmed by direct declaration-order adjacency in `ram/wram.asm` (`wYCoord`,
  `wXCoord` declared immediately after `wCurrentTileBlockMapViewPointer`, then
  `wYBlockCoord`/`wXBlockCoord` declared as a distinct pair right after).
- **`game_area_collision()`'s 2x2-doubled grid** (used by this repo's local A* in
  `navigation.py`) is a PyBoy/Game-Boy-screen-rendering artifact (`_walkable`'s
  grid is 18x20 raw tiles, doubled up from a 9x10 *block* grid), and is unrelated
  to `wXCoord`/`wYCoord`/`warp_event` units — `navigation.py` already accounts for
  this itself (`_PLAYER_GRID_COL`/`_PLAYER_GRID_ROW` = the fixed screen-relative
  cell PyBoy's collision grid always puts the player on) and nothing about warp
  data changes that; a warp/waypoint graph built from `warp_event` X/Y would
  operate in `player_x`/`player_y` world-tile units exactly like
  `NavigationTarget.x`/`.y` already do, only being converted to the on-screen
  collision grid's frame the same way `execute_navigation_macro` already converts
  its `target.x`/`.y` (`goal_col = _PLAYER_GRID_COL + dx`, etc.) — for the "last
  few tiles inside a single map" local-A* handoff #84 describes.

## 3. Completeness across this repo's 14 milestone maps

Every one of the 14 maps `milestones.py` currently targets has a populated
`def_warp_events` table in its `data/maps/objects/<MapName>.asm` file. Fetched and
inspected directly (map IDs cross-checked against `constants/map_constants.asm`,
matching `milestones.py`'s `_MAP_*` constants exactly):

| Milestone map (`milestones.py`) | Map ID | `pret/pokered` objects file | Warp entries (source order) |
|---|---|---|---|
| Oak's Lab | 40 | `OaksLab.asm` | 2: `(4,11)→LAST_MAP#3`, `(5,11)→LAST_MAP#3` |
| Viridian Mart | 42 | `ViridianMart.asm` | 2: `(3,7)→LAST_MAP#2`, `(4,7)→LAST_MAP#2` |
| Viridian Gym | 45 | `ViridianGym.asm` | 2: `(16,17)→LAST_MAP#5`, `(17,17)→LAST_MAP#5` |
| Pewter Gym | 54 | `PewterGym.asm` | 2: `(4,13)→LAST_MAP#3`, `(5,13)→LAST_MAP#3` |
| Cerulean Gym | 65 | `CeruleanGym.asm` | 2: `(4,13)→LAST_MAP#4`, `(5,13)→LAST_MAP#4` |
| Bill's House | 88 | `BillsHouse.asm` | 2: `(2,7)→LAST_MAP#1`, `(3,7)→LAST_MAP#1` |
| Vermilion Gym | 92 | `VermilionGym.asm` | 2: `(4,17)→LAST_MAP#4`, `(5,17)→LAST_MAP#4` |
| Champion's Room | 120 | `ChampionsRoom.asm` | 4: `(3,7)→LANCES_ROOM#2`, `(4,7)→LANCES_ROOM#3`, `(3,0)→HALL_OF_FAME#1`, `(4,0)→HALL_OF_FAME#1` |
| Celadon Gym | 134 | `CeladonGym.asm` | 2: `(4,17)→LAST_MAP#7`, `(5,17)→LAST_MAP#7` |
| Pokemon Tower 7F | 148 | `PokemonTower7F.asm` | 1: `(9,16)→POKEMON_TOWER_6F#2` |
| Fuchsia Gym | 157 | `FuchsiaGym.asm` | 2: `(4,17)→LAST_MAP#6`, `(5,17)→LAST_MAP#6` |
| Cinnabar Gym | 166 | `CinnabarGym.asm` | 2: `(16,17)→LAST_MAP#2`, `(17,17)→LAST_MAP#2` |
| Saffron Gym | 178 | `SaffronGym.asm` | 32: 2 normal entrance warps (`(8,17)`/`(9,17)→LAST_MAP#3`) + 30 same-map teleport-tile "warps" (`dest_map = SAFFRON_GYM`, various indices) |

Observations from this pass:

- **No gaps.** All 14 maps have real, non-empty warp tables — this data source is
  complete for the milestone list as it stands today.
- **Champion's Room is a genuine multi-exit case**, not just an entrance/exit
  pair: two warps arrive from `LANCES_ROOM` (the player enters after beating the
  Elite Four), and two more lead forward to `HALL_OF_FAME` after winning — useful
  to know if a future milestone wants a post-victory destination too.
  (https://github.com/pret/pokered/blob/master/data/maps/objects/ChampionsRoom.asm)
- **Saffron Gym's teleport-tile puzzle is itself expressed entirely as
  `warp_event` entries** whose destination map is Saffron Gym itself — confirming
  the format handles same-map "warps" (a case #84's waypoint-graph design should
  account for: not every `dest_map` is a *different* map) with no special-casing
  needed in the data format, only in graph-building logic that must not assume
  every warp edge leaves the current map.
  (https://github.com/pret/pokered/blob/master/data/maps/objects/SaffronGym.asm)
- **`LAST_MAP` (destination `-1`/`$ff`) appears on 11 of the 14 maps'** simple
  entrance/exit doors — meaning a static waypoint graph can't hardcode those
  warps' destination map from the source line alone; it resolves dynamically at
  runtime to whatever map the player warped in from (`wLastMap`). A graph built
  from this data would need to special-case `LAST_MAP` edges as "return to
  previous map" rather than a fixed edge to one specific map, or resolve them
  per-instance from the *entering* map's own matching exit warp (both of the
  milestone maps' `LAST_MAP` doors are themselves two-tile-wide entries mirroring
  a corresponding outdoor-map door — e.g. Pewter Gym's `(4,13)`/`(5,13)` pair
  likely corresponds 1:1 to Pewter City's own gym-door warp entering `PEWTER_GYM`
  at warp index 3 — but confirming that requires cross-referencing the *outdoor*
  maps' own objects files, out of scope for this pass since none of the 14
  milestones is an outdoor overworld map itself).
- **Two milestone maps show non-`LAST_MAP` explicit destinations**
  (`PokemonTower7F.asm`'s single warp back down to `POKEMON_TOWER_6F`, and
  `ChampionsRoom.asm`'s warps to `LANCES_ROOM`/`HALL_OF_FAME`) — these are directly
  and statically resolvable without runtime `wLastMap` ambiguity, and are the
  simplest case for a hand-authored graph.

## Open follow-ups for whoever builds #84's graph

- This pass only fetched the 14 milestone maps' own `objects` files plus the
  macro/runtime-code sources needed to establish format and coordinate frame — it
  did not fetch or diagram the full ~200-map graph, nor the outdoor overworld
  maps (Pallet Town, Viridian City, etc.) whose own doors are the *other* end of
  each `LAST_MAP` warp above. Building the actual waypoint graph will need those
  outdoor maps' `objects` files too (for both their own `warp_event` entries and,
  where routes touch, their headers' `connection` entries), plus a resolution
  strategy for `LAST_MAP` edges (see previous section).
- None of this was cross-checked against a live/dumped compiled ROM's WRAM in
  this pass (this was a disassembly-source-and-runtime-code reading exercise, not
  a PyBoy boot-and-verify pass, matching this ticket's "primary sources" scope) —
  worth a follow-up spot-check the way `tests/test_navigation.py` already spot-checks
  `player_x`/`player_y` and the collision grid against a real boot, before any
  graph built from this data is trusted for navigation without further
  verification.
