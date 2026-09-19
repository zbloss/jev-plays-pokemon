# Research: Gen 1 map/warp coordinate data is machine-readable, not human-documented

Informs [#84](https://github.com/zbloss/jev-plays-pokemon/issues/84) (tile-level milestone
targets + cross-map travel) and [#83](https://github.com/zbloss/jev-plays-pokemon/issues/83)
(the stop-gap). Supersedes nothing; complements
[`pokemon-red-ram-map.md`](./pokemon-red-ram-map.md), which covers *reading* state and this
note covers *deriving* world geometry.

## Question

#84 states that tile-level destination coordinates "are not in the documentation", that each
value "has to be produced *and* confirmed by booting the ROM", that the pinned PyBoy "doesn't
surface the WRAM-buffered full-map block data", and that warp tiles force a hand-authored
semantics decision. Is there machine-readable map/coordinate data — online or inside the game
data — that this repo can parse or generate from instead of hand-documenting tile by tile?

## Top-line answer

**Yes, and it is already in this repo's own `pokemon_red.gb`.** Every quantity #84 calls manual
human work is a byte in the ROM, addressable through tables the ROM itself publishes:

- **Tile-level NPC destinations** are object-event records. `pret/pokered` writes them as plain
  text (`object_event 4, 1, SPRITE_SUPER_NERD, STAY, DOWN, TEXT_PEWTERGYM_BROCK, OPP_BROCK, 1`),
  and the repo's ROM contains the compiled equivalents. **11 of 14** milestone targets came out
  of a 200-line parser in the first pass; the last **3** were a lookup bug in the scratch script,
  not missing data, and are resolved below — so all 14 are available now.
- **The coordinate frame already matches.** `pret/pokered`'s own overworld engine compares
  `[wXCoord]`/`[wYCoord]` byte-for-byte against map event coordinates at runtime
  (`home/overworld.asm:391-408`). Those are the same `0xD361`/`0xD362` this repo reads in
  `game_state.py:101-103`. No transform is needed.
- **Cross-map travel is a 1244-edge graph**, already extracted, plus 86 map-connection records.
- **The whole map's block grid is in WRAM at `$C6E8`** — the game itself loads the *complete*
  map there on every map load (`home/overworld.asm:882`), and it was read live out of
  `pyboy.memory` and matched the upstream `.blk` bytes exactly.

The one thing that survives #84's argument is the **verification discipline**, and its reason
changed: coordinates can be *generated*, but a given NPC's live position depends on script state
(Oak has two positions; `WALK` objects move; doors open on flags), so a booting-the-ROM check is
still required — as a cheap assertion per entry, not as the discovery mechanism #84 assumes.

**What does not exist yet and is genuinely open:** turning a map's block grid into a walkability
grid. Upstream supplies every ingredient (block grids, `gfx/blocksets/*.bst`, per-tileset
passable-tile lists in `data/tilesets/collision_tile_ids.asm`) and this note confirms the pieces
are machine-readable, but nobody here has wired them together, and which of a block's four
sub-tiles the engine tests depends on direction of entry. That is real work — bounded, and
~25 tilesets deep, not 14 milestones deep.

## Verdict on #84's three claims

| #84 claim | Verdict | Evidence |
|---|---|---|
| "The coordinates are not in the documentation… not 'where Brock stands in Pewter Gym' in the coordinate frame this repo's own RAM reads use." | **Refuted** | `pret/pokered data/maps/objects/PewterGym.asm` literally states it: `object_event 4, 1, SPRITE_SUPER_NERD, …, TEXT_PEWTERGYM_BROCK, OPP_BROCK, 1`. Same frame is proven by the engine's own compare at `home/overworld.asm:391-408`. 1466/1466 object records across 223 maps decode to `y+4`/`x+4` with zero exceptions. |
| "The pinned PyBoy version caps the routing model… doesn't surface the WRAM-buffered full-map block data either." | **Refuted** | `home/overworld.asm:882`: "; this loads the current map's complete tile map (which references blocks, not individual tiles) to `wOverworldMap`". `ram/wram.asm:191` `wOverworldMap:: ds 1300`. Live read of `pyboy.memory[0xC6E8]` with stride `width+6` reproduced Pallet Town's whole block grid, byte-identical to `maps/PalletTown.blk` (`live_check2.json`: `"all_rows_match_blk": true`). The API gap is real about *PyBoy's convenience wrapper*; WRAM is flatly readable, so it was never a blocker. |
| "Warp tiles are a correctness trap… any design here must decide explicitly whether warp tiles are obstacles, deliberate transitions, or context-dependent." | **Partially refuted** | *Identification* is data: per-tileset warp/door tile ID lists (`data/tilesets/warp_tile_ids.asm`, `door_tile_ids.asm`, `-1`-terminated) and per-map warp records with real destinations. The **semantics** decision is still a design call — but it is now a decision over enumerated tiles with known destinations, not over unknown ones. |

## Sources consulted (primary)

Cloned locally for byte-level work; both pinned:

1. **`pret/pokered`** @ `a1a22aaf84d1675bcdbaeb194592379d586d838e` (2026-08-27) →
   `.qwen/tmp/pokered`. Canonical disassembly. Files cited below by path + line.
2. **This repo's `pokemon_red.gb`** (working tree) — the artifact under test, parsed directly.
3. **`PWhiddy/PokemonRedExperiments`** @ `5fa47ef0575f35a22b1b50367b7f317bcf815747` →
   `.qwen/tmp/PokemonRedExperiments`. Independent derived datasets (see §7).
4. **This repo's own code**, as the coordinate-frame reference:
   `src/jev_plays_pokemon/game_state.py:101-103`, `navigation.py:44-62`.

Nothing in this note rests on a wiki or a secondary write-up.

## 1. Foundation: the repo's ROM *is* pokered's data, map-for-map

The parser resolves map id → (bank, address) through the ROM's own
`MapHeaderPointers`/`MapHeaderBanks` tables (`map_pointer_tables.json`: pointer table at file
offset `430`, bank table at `49725`, located empirically by `.blk`-anchoring), then validates
against upstream rather than trusting the disassembly's symbolic layout:

```text
maps in table: 223 | block grids byte-identical to pokered maps/*.blk: 216
record sets matching pokered data/maps/objects/*.asm: 218
warp records: 1244 | map-connection records: 86
```

Unmatched: `SaffronCity`, `LancesRoom`, `Route16Gate1F`, `RocketHideoutElevator`, `SilphCo2F`
(blocks *and* records), plus `UndergroundPathNorthSouth` (blocks) and
`UndergroundPathRoute7Copy` (upstream `.blk` is empty). Upstream *has* the corresponding
`maps/*.blk` and `data/maps/objects/*.asm` files, so these are unresolved parser disagreements —
most likely a map-id slot assignment or a border-block byte — **not** missing upstream data.
**None of the 14 milestone maps is in the failing set.** Treat the 7 as "verify before use", and
the 5 record mismatches as a real open item if any of them ever becomes a waypoint.

## 2. Tile-level targets: all 14 milestones, in the repo's own frame

From `.qwen/tmp/rom_maps_full.json` (via the ROM's own tables) cross-checked against
`data/maps/objects/<Map>.asm`. Coordinates are tile coordinates, `x` right, `y` down — the same
frame as `GameState.player_x`/`player_y`.

| Milestone | Map | `map_id` | Target const | x | y |
|---|---|---|---|---|---|
| `got_starter` | OaksLab | 40 | `OAKSLAB_CHARMANDER_POKE_BALL` | 6 | 3 |
| `got_starter` (alt) | OaksLab | 40 | `OAKSLAB_SQUIRTLE_POKE_BALL` / `_BULBASAUR_POKE_BALL` | 7 / 8 | 3 |
| `got_oaks_parcel` | ViridianMart | 42 | `VIRIDIANMART_COOLTRAINER_M` | 3 | 3 |
| `got_pokedex` | OaksLab | 40 | `OAKSLAB_RIVAL` | 4 | 3 |
| `boulder_badge` | PewterGym | 54 | `PEWTERGYM_BROCK` | 4 | 1 |
| `cascade_badge` | CeruleanGym | 65 | `CERULEANGYM_MISTY` | 4 | 2 |
| `got_ss_ticket` | BillsHouse | 88 | `BILLSHOUSE_BILL1` | 4 | 4 |
| `thunder_badge` | VermilionGym | 92 | `VERMILIONGYM_LT_SURGE` | 5 | 1 |
| `rainbow_badge` | CeladonGym | 134 | `CELADONGYM_ERIKA` | 4 | 3 |
| `got_poke_flute` | PokemonTower7F | 148 | `POKEMONTOWER7F_MR_FUJI` | 10 | 3 |
| `soul_badge` | FuchsiaGym | 157 | `FUCHSIAGYM_KOGA` | 4 | 10 |
| `marsh_badge` | SaffronGym | 178 | `SAFFRONGYM_SABRINA` | 9 | 8 |
| `volcano_badge` | CinnabarGym | 166 | `CINNABARGYM_BLAINE` | 3 | 3 |
| `earth_badge` | ViridianGym | 45 | `VIRIDIANGYM_GIOVANNI` | 2 | 1 |
| `beat_champion` | ChampionsRoom | 120 | `CHAMPIONSROOM_RIVAL` | 4 | 2 |

`got_poke_flute`, `marsh_badge` and `beat_champion` are the three `milestone_targets.json`
labelled `"error": "not derived"`; they are present and validated in
`rom_maps_full.json` (`blocks_match_blk_file: true`, `records_match_asm: true`), so the earlier
failure was the scratch script looking them up in its own weaker `.blk`-anchored table.

Two honesty notes on that table:

- **The object *choice* is authored, not derived.** The scratch script picked a target by
  substring match (`who in name_const`), and `"OAK"` matches every `OAKSLAB_*` constant — so
  `got_starter` resolved to `OAKSLAB_RIVAL` rather than the starter ball. Populating
  `Milestone.target_x/target_y` needs a deliberate 14-line map id→const choice. The *coordinates*
  are data; the *mapping* is judgment (and `got_starter`'s target is genuinely dynamic — which of
  three balls depends on the player's pick).
- **Positions are script-stateful.** `OAKSLAB_OAK1` (5,2) and `OAKSLAB_OAK2` (5,10) are the same
  trainer at two story beats; `PALLETTOWN_OAK` (8,5) walks into the lab; `WALK` objects
  (`PALLETTOWN_GIRL`, `_FISHER`) roam. Pick the candidate for the state the milestone implies,
  then confirm on boot (§ "what still needs emulator work").

### The compiled record encoding (verified, 1466/1466)

Object records are 6 bytes: `[sprite, y + 4, x + 4, 0xFF, flag_hi, flag_lo]`. Across **all 223
derived maps, all 1466 object records**, `raw[0] == sprite` and `raw[1] == y + 4` and
`raw[2] == x + 4`, with zero exceptions:

```text
objects with raw: 1466 offset+4 holds: 1466 fails: 0
```

Do **not** read object coordinates straight off bytes 1/2 as tiles — that off-by-four is exactly
the kind of thing that produces a "verified" coordinate nobody can reproduce at the keyboard.
Prefer parsing pokered's `data/maps/objects/*.asm`, where the coordinates are explicit
`object_event x, y, …` arguments.

## 3. Coordinate frame: proven by the engine, not by analogy

`pret/pokered ram/wram.asm`:

```asm
1782: wCurMap:: db
1788: wYCoord:: db
1789: wXCoord:: db
1829: wWarpEntries:: ds MAX_WARP_EVENTS * 4 ; Y, X, warp ID, map ID
```

`pret/pokered home/overworld.asm:391-408` (`CheckWarpsNoCollision`) loads `[wYCoord]`→`d`,
`[wXCoord]`→`e` and compares them directly against each `wWarpEntries` Y/X pair — if event and
player coordinates were different frames, the game's own door logic would not work:

```asm
	ld a, [wYCoord]
	ld d, a
	ld a, [wXCoord]
	ld e, a
	ld hl, wWarpEntries
CheckWarpsNoCollisionLoop::
	ld a, [hli] ; check if the warp's Y position matches
	cp d
```

This repo reads the same three addresses: `game_state.py:101-103`
(`_MAP_ID_ADDRESS = 0xD35E`, `_PLAYER_Y_ADDRESS = 0xD361`, `_PLAYER_X_ADDRESS = 0xD362`).

**Live confirmation, not just source reading.** Booting this repo's ROM fresh, walking out of
Red's House with `tests/test_navigation.py`'s existing path, then reading `wNumberOfWarps`
(`0xD3AE`) and `wWarpEntries` (`0xD3AF`) gave Pallet Town's three warps:

| Live from RAM | pokered `data/maps/objects/PalletTown.asm` |
|---|---|
| `(x5,y5) → map 37, warp 0` | `warp_event 5, 5, REDS_HOUSE_1F, 1` |
| `(x13,y5) → map 39, warp 0` | `warp_event 13, 5, BLUES_HOUSE, 1` |
| `(x12,y11) → map 40, warp 1` | `warp_event 12, 11, OAKS_LAB, 2` |

Coordinates match exactly; the only offset is the warp index being 1-based upstream, 0-based in
RAM. `dest_map == 255` (`$FF`) is pokered's `LAST_MAP` sentinel, resolved to the current map at
load time — decode it, don't treat 255 as a map.

## 4. Full-map geometry is in WRAM (and in the ROM, statically)

Two independent routes, both usable:

**Statically, no emulator.** `derive_maps.py` walks map header → block pointer → file offset and
confirms the bytes equal pokered's `maps/<Map>.blk`. Header layout, all bank-relative, from
`macros/scripts/maps.asm`:

```text
+0 tileset id | +1 height_blocks | +2 width_blocks | +3 dw blocks | +5 dw texts
+7 dw script  | +9 connection mask (EAST=1 WEST=2 SOUTH=4 NORTH=8)
+10 one 11-byte connection record per set bit, in EAST,WEST,SOUTH,NORTH order
+10+11n dw objects
```

A block is 2×2 tiles, so `width_tiles = width_blocks * 2`. Pallet Town: 10×9 blocks = 20×18
tiles; Pewter Gym: 5×7 blocks = 10×14 tiles, tileset 7, with `PEWTERGYM_BROCK` at (4,1) inside it.

**At runtime, on the current map.** `home/overworld.asm:882` documents that the routine loads
"the current map's **complete** tile map (which references blocks, not individual tiles) to
`wOverworldMap`". `macros/coords.asm:72` gives the exact indexing math, confirming the
`width + 6` stride with a 3-block border that `live_check2.py` found empirically:

```asm
	ld \1, wOverworldMap + ((\2) + 3) + (((\3) + 3) * ((\4) + (3 * 2)))
```

Live read: stride 16 for Pallet Town, origin `0xC71B`, `all_rows_match_blk: true`, row 3
`4e 01 01 01 01 01 01 01 01 4d` identical in WRAM and in `maps/PalletTown.blk`. So global routing
over the *current* map needs no PyBoy feature at all; and for *other* maps, the static route
covers it. `navigation.py:44-52`'s "only the scrolled window is exposed" framing is about
`game_area_collision`, and is worth correcting in that docstring.

## 5. Warps and connections: the graph is already there

`.qwen/tmp/warp_edges.json`: **1244** edges, each
`{src_map, src_map_name, warp_index_0based, tile_x, tile_y, dest_map, dest_warp_0based}`.
Destination tile resolves by looking up `dest_map`'s own warp list at `dest_warp_0based` —
verified for Pallet Town → Oak's Lab: door record `(12,11) → map 40, warp 1`, and map 40's warp 1
is `(5,11)`, exactly the arrival prediction the script computed.

86 connection records handle walk-between-maps continuity (route ↔ route, no door), each carrying
destination map, block length, and the `wOverworldMap` window pointer to patch.

Warp/door *tile IDs* per tileset are explicit lists in
`data/tilesets/warp_tile_ids.asm` / `door_tile_ids.asm` (`WarpTileIDPointers`, `-1`-terminated).
That is what makes #84's warp-trap concern tractable: the router can treat a warp tile as
"obstacle unless this edge is the chosen route", and it can know where each tile leads before
stepping on it.

## 6. Walkability: the real gap

Solved inputs: block grids (§4), block→tile mapping in `gfx/blocksets/*.bst` (`gfx/tilesets.asm:4`
`Overworld_Block:: INCBIN "gfx/blocksets/overworld.bst"`; `gym.bst` is 1856 bytes,
`overworld.bst` 2048), and per-tileset passable-tile lists in
`data/tilesets/collision_tile_ids.asm`:

```asm
Overworld_Coll::
	coll_tiles $00, $10, $1b, $20, $21, $23, $2c, $2d, $2e, $30, $31, $33, $39, $3c, $3e, $52, $54, $58, $5b
```

consumed through `wTilesetCollisionPtr` (`ram/wram.asm:1885`; `home/overworld.asm:1263`,
`:1910`, annotated "; pointer to list of passable tiles").

**Not done:** wiring those three together into a per-map walkability grid, and pinning down which
of a block's four sub-tiles the engine tests as a function of entry direction
(`CheckTilePassable`, `home/overworld.asm:1259`, called from `:1242`). Until that is right, a global route can plan
through a wall or refuse a legal ledge. This is the honest remaining engineering work, and it is
per-tileset (~25), not per-milestone. `UNVERIFIED`: the exact sub-tile selection rule — read
`home/overworld.asm`'s collision path before writing the router.

## 7. Existing derived datasets (prior art, verified on disk)

`PWhiddy/PokemonRedExperiments/baselines/` ships ready-made data. The repo already trusts this
project: `game_state.py:10` cross-checks its addresses against `memory_addresses.py` there.

- **`map_data.json`** (58 KB) — `"regions"`: per map id, `name`, global `coordinates`,
  `tileSize`. Pallet Town is `id "0"`, `tileSize [20, 18]` — matching this repo's derived
  20×18 tiles independently. Useful for a global-map view; it is *not* a warp graph and carries
  no per-tile collision.
- **`global_map.py`** (1.1 KB) — local→global tile transform over `map_data.json`. Its header
  attributes the dataset upstream: `adapted from
  https://github.com/thatguy11325/pokemonred_puffer/blob/main/pokemonred_puffer/global_map.py`.
- **`events.json`** (66 KB) — event-flag bit names keyed by `0xD747-n` … i.e. the same range as
  this repo's `_EVENT_FLAGS_START_ADDRESS = 0xD747` (`game_state.py:70`), with readable labels
  ("Followed Oak Into Lab", "Got Town Map", "Pallet After Getting Pokeballs"). A shortcut for
  milestone-completion conditions.
- **`PWhiddy/pokegb`** is a C Game Boy core, unrelated to map data (checked, ruled out).

Dead-end detail worth keeping: no committed JSON/CSV of *milestone tile targets* exists anywhere
checked — the datasets above give geometry, global layout, and flags. The 14-entry object choice
is the only piece with no upstream equivalent, and it is 14 lines.

## Recommended pipeline

1. **Vendor a generator, not a CSV.** Pin `pret/pokered` at a commit and generate
   `map_id / target_x / target_y` + warp edges at test/build time from
   `data/maps/objects/*.asm` and `maps/*.blk`, or from the ROM's own tables (the §1 parser needs
   only `pokemon_red.gb`, which the project already requires). `pret/pokered` ships **no LICENSE
   file** (it is a disassembly of proprietary firmware), so a generator pinned to a commit is the
   safer shape than committing a derived dataset; confirm with the repo's own licensing stance
   before committing anything derived.
2. **Author the 14-line milestone→object-const mapping deliberately** (see the `OAK` substring
   trap), flagging dynamic targets (`got_starter`) rather than pretending to one tile.
3. **Populate `Milestone.target_x/target_y`** from the generated table;
   `resolve_navigation_target` (`navigation.py:211-223`) then stops returning `None` — #84's
   second acceptance criterion, with no hand-documentation.
4. **Build the warp graph** from `warp_edges.json`-equivalent output + the 86 connection records;
   route map-to-map over that graph, then let today's local A\* handle tiles inside one map.
   Record the warp-tile semantics here, as #84 asks, but as a decision over enumerated tiles.
5. **Solve block→walkability** (§6) per tileset.
6. **Keep the boot check, demote it to an assertion.** For each milestone: set the flags the
   milestone presumes, boot, read `wXCoord`/`wYCoord`/`wCurMap`, and assert the target is
   reachable and that the NPC's live tile matches the generated candidate. That is a per-entry
   test, minutes to run — not per-entry manual documentation.

## What still genuinely needs human/emulator work

- Block→walkability semantics per tileset, including direction-dependent sub-tile tests (§6).
- Script-gated reachability: closed doors, boulders, suds, item blocks, `SilphCo` keys. The data
  says a tile is walkable; it does not say the tile is walkable *at your story state*. #84 scopes
  this out, correctly.
- The 5 maps whose ROM records disagree with upstream `.asm` (§1) — verify before use.
- Choosing which object *is* the objective, and what to do when it moves.
- `execute_navigation_macro` reaching the door tile at all: in the live run the macro was told to
  reach Pallet Town (12,11) and ended at (5,8) — it moved (`macro_moved: true`) but never arrived,
  so **the cross-map traversal end-to-end is still undemonstrated**. #84's first acceptance
  criterion (a ROM-backed test ending on the target's map) is not met by this research; the data
  is, the routing is not.

## What is NOT available / dead ends

- No upstream file enumerates milestone→tile targets. Not in pokered, not in the RL datasets.
- `PWhiddy/pokegb`: unrelated (C emulator core).
- pokered's `tools/` contains assembler/graphics tooling, no map-JSON exporter.
- The pinned PyBoy Gen1 wrapper does **not** grow a full-map API; the WRAM route replaced it.
- `UndergroundPathRoute7Copy`'s upstream `.blk` is empty (`blk_len 0`) — a duplicate map slot, not
  a usable map.

## Reproducing this

Scratch-only, in `.qwen/tmp/` (not project code, deliberately not under `src/` or `tests/`):

| File | Role |
|---|---|
| `parse_rom_maps.py`, `parse_rom_maps2.py` | `.blk`-anchored map identification, pokered const resolution (`const_def`/`const_skip`/`map_const`), milestone table |
| `derive_maps.py` | resolves every map through the ROM's own header/bank tables; validates dims/records/blocks against pokered; emits `rom_maps_full.json`, `warp_edges.json` |
| `live_check.py`, `live_check2.py` | boots this repo's `pokemon_red.gb` via the project's own `emulator`/`navigation` modules; reads `wWarpEntries` and `wOverworldMap` live; compares to static |
| `close_gaps.py` | re-resolves the 3 "not derived" milestones against `rom_maps_full.json` |

```cmd
uv run python .qwen\tmp\derive_maps.py
uv run python .qwen\tmp\live_check2.py
uv run python .qwen\tmp\close_gaps.py
```

`live_check2.py` writes no save state and does not modify the ROM. Clones live at
`.qwen/tmp/pokered` (`a1a22aa`) and `.qwen/tmp/PokemonRedExperiments` (`5fa47ef`); both are
re-fetchable at those commits.
