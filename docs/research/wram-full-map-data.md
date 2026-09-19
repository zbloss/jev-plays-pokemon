# Research: Full-map block/collision data in Gen 1 WRAM

Resolves issue [#86](https://github.com/zbloss/jev-plays-pokemon/issues/86), child of the wayfinder-map
issue [#85](https://github.com/zbloss/jev-plays-pokemon/issues/85), itself supporting the global-routing
candidate architecture proposed in [#84](https://github.com/zbloss/jev-plays-pokemon/issues/84).

## Question

Does Gen 1 Pokemon Red's WRAM expose full-map block/collision data (not just the on-screen-scrolled
window that PyBoy 2.2.0's `game_area_collision` returns), in a way readable via raw memory access
(`pyboy.memory[address]`) the same way `src/jev_plays_pokemon/game_state.py` already reads player
position/event flags/badges directly by address?

## Top-line answer

**Yes to all three parts.** Gen 1's WRAM does hold a full-current-map block buffer — `wOverworldMap`,
at `0xC6E8`–`0xCBFC` (1300 bytes) — populated once per map load with the *entire* current map's
block-ID layout (not just what's on screen), and it is **already reachable today, at the pinned
PyBoy 2.2.0, via a plain `pyboy.memory[...]` read**, with no version bump required — the same generic,
unrestricted memory accessor this repo's `game_state.py` already uses for `wYCoord`/`wObtainedBadges`/etc.
covers this address too. The buffer holds *block IDs*, not literal walkability bits, so getting
"collision" out of it (rather than just "layout") still requires the same block→tile→walkable-list
decode PyBoy's own screen-window `game_area_collision` already performs — just applied over the whole
buffer instead of the visible slice. A PyBoy version bump would not hand this project that decode step
for free: diffing 2.2.0's Gen1 wrapper against PyBoy's current `master` shows the collision-derivation
code is byte-for-byte unchanged there — the project would still have to build it itself either way.

## Sources consulted (primary)

1. **`pret/pokered`** — canonical disassembly of Pokemon Red/Blue, read directly (not a secondary
   write-up):
   - `ram/wram.asm`: https://github.com/pret/pokered/blob/master/ram/wram.asm
   - `home/overworld.asm` (map-loading code, `LoadTileBlockMap`):
     https://github.com/pret/pokered/blob/master/home/overworld.asm
   - `engine/overworld/update_map.asm` (`ReplaceTileBlock`, which indexes `wOverworldMap` at runtime):
     https://github.com/pret/pokered/blob/master/engine/overworld/update_map.asm
   - `constants/gfx_constants.asm` (`BLOCK_WIDTH`/`SCREEN_BLOCK_WIDTH`/`SURROUNDING_WIDTH` etc.):
     https://github.com/pret/pokered/blob/master/constants/gfx_constants.asm
   - `constants/map_data_constants.asm` (`MAP_BORDER`):
     https://github.com/pret/pokered/blob/master/constants/map_data_constants.asm
   - `layout.link` (confirms WRAM0 origin `$C100`, matching this repo's own derivation method):
     https://github.com/pret/pokered/blob/master/layout.link
   - **`pokered.sym`**, the linker-generated symbol table on the repo's own `symbols` branch (built
     directly from `master` by the project's CI, so it is the actual computed addresses, not a
     hand-summed reconstruction): https://github.com/pret/pokered/blob/symbols/pokered.sym — used to
     get exact addresses instead of hand-summing every preceding WRAM declaration, and cross-checked
     against every address `game_state.py` already independently verified (see below).
2. **PyBoy** (`Baekalfen/PyBoy`) — read at the exact pinned tag and at `master`, not a docs paraphrase:
   - Pinned version's memory accessor source (`PyBoyMemoryView`):
     https://github.com/Baekalfen/PyBoy/blob/v2.2.0/pyboy/pyboy.py
   - Pinned version's Gen 1 game wrapper (`game_area_collision`/`_get_screen_walkable_matrix`):
     https://github.com/Baekalfen/PyBoy/blob/v2.2.0/pyboy/plugins/game_wrapper_pokemon_gen1.py
   - Current `master`'s Gen 1 game wrapper, diffed against the above:
     https://github.com/Baekalfen/PyBoy/blob/master/pyboy/plugins/game_wrapper_pokemon_gen1.py
   - GitHub Releases (`v2.2.1` through `v2.7.1`), read via `gh api repos/Baekalfen/PyBoy/releases/tags/<tag>`:
     https://github.com/Baekalfen/PyBoy/releases
3. This repo's own already-verified addresses (`src/jev_plays_pokemon/game_state.py`), used only as a
   cross-check, not as a new source of truth.

## Finding 1 — `wOverworldMap`: a genuine full-current-map buffer exists in WRAM

`pret/pokered`'s `ram/wram.asm` declares (line ~191):

```
SECTION "Overworld Map", WRAM0
; This union spans 1300 bytes.
UNION
wOverworldMap:: ds 1300
wOverworldMapEnd::
NEXTU
wTempPic:: ds PIC_SIZE tiles
ENDU
```

Per `pokered.sym` (the disassembly's own build output): **`wOverworldMap` = `0xC6E8`, ending at
`0xCBFC`** (`0xCBFC − 0xC6E8 = 0x514 = 1300` bytes, confirming the declared size). This sits inside
WRAM0 (`$C000`–`$CFFF`, non-banked on DMG), directly adjacent to `wSurroundingTiles` (`0xC508`, ending
exactly at `0xC6E8` where `wOverworldMap` begins — confirming the source's declared union/section
ordering).

**This is confirmed to be the *entire current map*, not a local window**, by reading the routine that
populates it, `LoadTileBlockMap` in `home/overworld.asm`, whose own comment states:

> "this loads the current map's complete tile map (which references blocks, not individual tiles) to
> `wOverworldMap`; it can also load partial tile maps of connected maps into a border of length 3
> around the current map"

Mechanically: it copies `wCurMapHeight` (`0xD368`) rows of `wCurMapWidth` (`0xD369`) bytes each,
straight from the map's ROM data pointer (`wCurMapDataPtr`), into `wOverworldMap` at a stride of
`wCurMapWidth + MAP_BORDER*2` (`MAP_BORDER = 3`, from `constants/map_data_constants.asm`) — i.e. it is
indexed using the map's *real, full* dimensions, not the screen's. The extra 3-block border on each
edge is then filled with the adjoining maps' edge strips (`wNorthConnectedMap`/`wSouthConnectedMap`/
etc.), so a player standing near a map's edge can still see one block into the next map. This same
buffer is written back into at runtime too — `engine/overworld/update_map.asm`'s `ReplaceTileBlock`
indexes into `wOverworldMap` using `wCurMapWidth` the same way, e.g. when a warp tile or cuttable tree
is dynamically edited — reinforcing that the buffer really is addressed as one full map, not a
scroll-relative window.

Each byte is a **block ID**, not a tile ID or a collision bit: Gen 1 maps are built from 4×4-tile
"blocks" (`BLOCK_WIDTH = BLOCK_HEIGHT = 4`, `constants/gfx_constants.asm`). This is a materially
different (and larger-scope) buffer than the two window-scoped ones also declared in `wram.asm`:

| Symbol | Address | Size | Scope |
|---|---|---|---|
| `wTileMap` | `0xC3A0` (per `wram.asm`'s section order) | `SCREEN_AREA` = 20×18 tiles | the literal on-screen VRAM-mirrored tilemap |
| `wSurroundingTiles` | `0xC508` | `SURROUNDING_WIDTH × SURROUNDING_HEIGHT` = 24×20 tiles (6×5 blocks) | a scroll-margin buffer *around the player*, still local, not full-map (`SCREEN_BLOCK_WIDTH=6`/`SCREEN_BLOCK_HEIGHT=5`, `gfx_constants.asm`) |
| `wOverworldMap` | `0xC6E8` | 1300 bytes, laid out as `(wCurMapWidth+6) × (wCurMapHeight+6)` blocks | **the whole current map's block IDs**, plus a 3-block connected-map border |

**Scope caveat:** "full map" here means the one map currently loaded (re-populated by `LoadTileBlockMap`
on every map transition) — matching this repo's existing per-turn `state.map_id` model in
`game_state.py` — not the entire game world in one buffer. That is consistent with what issue #84/#85
actually need (routing within/across individually-loaded maps), not a single monolithic world atlas.
The 1300-byte cap is also pokered's own de facto limit on how large any single map can legally be
(stride × row-count must fit in 1300 bytes), so for any map the ROM can actually load, this buffer
already holds it in full — there is no larger "still windowed" map data hiding elsewhere in WRAM for a
legally-sized map.

`wOverworldMap` gives block **layout**, not walkability directly. Turning it into a collision grid
needs the same decode PyBoy's own screen-window collision logic already performs (see Finding 2): each
block ID must be expanded to its 16 constituent tile IDs via the current tileset's blockset data
(`wTilesetBlocksPtr`, `0xD52C`, a ROM pointer) and each of those tile IDs checked against the current
tileset's walkable-tile list (`wTilesetCollisionPtr`, `0xD530`) plus the grass-tile special case
(`wGrassTile`, `0xD535`) and tileset type (`0xFFD7`) — this project would still have to write that
block→tile→walkable expansion; `wOverworldMap` supplies the missing *layout* input to it, not a
finished collision grid.

## Finding 2 — Reachable at the pinned PyBoy 2.2.0 via `pyboy.memory[]`, no version bump needed

PyBoy 2.2.0's `PyBoyMemoryView.__getitem__` (`pyboy/pyboy.py`) is a **generic, address-driven** Game
Boy memory accessor, not one scoped to whatever the active game wrapper happens to expose. For an
unbanked read (`pyboy.memory[addr]`, no explicit bank tuple — the same call form `game_state.py`
already uses for every field it reads), the implementation falls straight through to
`self.mb.getitem(x)`, the motherboard's general memory-map dispatch covering the full 16-bit Game Boy
address space (boot ROM, cartridge ROM banks, VRAM, cartridge RAM banks, WRAM, echo RAM, OAM, I/O
registers, HRAM) — it does not special-case or restrict which WRAM addresses are legal to read. The
same docstring in that file explicitly documents this as the way to reach RAM the higher-level wrapper
API doesn't expose ("The find addresses of interest, either search online for something like: '[game
title] RAM map'...").

`0xC6E8`–`0xCBFC` is squarely inside WRAM0 (`0xC000`–`0xCFFF`), the exact same class of address this
repo already reads without issue (e.g. `wIsInBattle` at `0xD057` is WRAM1, one region over, reached via
the identical code path). There is nothing about `wOverworldMap`'s address that makes it any less
reachable than the addresses `game_state.py` already relies on — `pyboy.memory[0xC6E8]` (a single byte)
or `pyboy.memory[0xC6E8:0xCBFC]` (the whole 1300-byte buffer as a list) work today, at 2.2.0, with the
exact same API surface this repo's `navigation.py`/`game_state.py` docstrings already describe as
"PyBoy's stable low-level APIs."

This is independently reinforced by reading PyBoy 2.2.0's own Gen1 wrapper source
(`game_wrapper_pokemon_gen1.py`): its `game_area_collision`/`_get_screen_walkable_matrix` never touch
`wOverworldMap` or `wCurMapWidth`/`wCurMapHeight` at all — they derive everything from
`_get_screen_background_tilemap`, which reads `pyboy.tilemap_background` (a VRAM-backed, scroll-rolled
20×18 view) and only the *screen-relative* bottom-left-of-each-2×2-tile sampling of it. This matches
(and independently confirms, by reading the actual pinned-version source rather than taking the
existing docstring's word for it) what `navigation.py`'s module docstring already states about the
API gap. Diffing this same method against PyBoy's current `master` (see Finding 3) shows it is
unchanged there too — so this is not a 2.2.0-specific limitation that a newer release quietly fixed.

`_get_screen_walkable_matrix`'s own collision-pointer read is worth noting as a second, independent
cross-check of the addresses above: it reads `self.pyboy.memory[0xD530]`/`self.pyboy.memory[0xD531]`
for the tileset's walkable-tile-list pointer and `self.pyboy.memory[0xD535]` for the grass tile — these
match `pokered.sym`'s `wTilesetCollisionPtr = 0xD530` and `wGrassTile = 0xD535` exactly, i.e. PyBoy's
own maintainer independently arrived at the same addresses this research pass got from the disassembly.

## Finding 3 — What a version bump would (and would not) buy

Diffing PyBoy 2.2.0's `game_wrapper_pokemon_gen1.py` against the current `master` branch shows `master`
adds a materially richer API surface — `party`/`money`/`inventory`/`set_badge`/`set_event_flag`/`warp`/
`start_wild_battle`/`start_trainer_battle`/text encode-decode — matching what this repo's
`game_state.py` docstring already described as living on "PyBoy's unreleased development branch."

**But `game_area_collision`/`_get_screen_walkable_matrix` are byte-for-byte unchanged** between 2.2.0
and today's `master` (only whitespace/formatting differs in the diff). PyBoy's own maintained codebase,
at its current HEAD, still has no full-map or `wOverworldMap`-aware collision API — it is still exactly
the same 20×18 on-screen-window derivation. **A version bump therefore would not hand this project a
ready-made full-map collision feature**; the block→tile→walkable-list expansion described in Finding 1
would have to be hand-written by this project regardless of which PyBoy version is pinned, using the
same raw-address technique that already works at 2.2.0.

Release notes relevant to a hypothetical bump (read via `gh api repos/Baekalfen/PyBoy/releases/tags/<tag>`,
i.e. GitHub's own release records, not a changelog write-up):

- **`v2.2.2`**: *"Fix address boundary issue in `pyboy.memory`"* — a patch-level bugfix directly in the
  accessor this approach would depend on; worth checking (2.2.0 vs. 2.2.2) if a raw-address read on the
  pinned version ever produces a suspicious off-by-one, though nothing in the code read here suggests
  it affects a plain in-bounds WRAM0 address like `wOverworldMap`.
- **`v2.4.0`**: *"CPU and memory timings have been corrected... If you depend on exact timing of a frame
  ... then expect these to have changed."* This is the one release-note item genuinely relevant to this
  repo beyond the map-data question: `navigation.py`'s `execute_button` is explicitly timing-sensitive
  (`_MAX_WALK_FRAMES`, per-tick position polling, documented at length in that module's own docstring),
  so bumping past `v2.4.0` would need that logic re-verified against a real boot, not assumed to still
  behave identically.
- **`v2.5.0`**: sound API overhaul, new `PyBoyException` hierarchy, Python 3.8 support dropped.
- **`v2.6.0`**: build/Cython pin changes, an LCD/sound double-speed crash fix.
- **`v2.7.0`/`v2.7.1`**: general compatibility and speed fixes; no memory-API or Gen1-wrapper changes
  mentioned.

None of these release notes describe any change to the *addressing model* of `pyboy.memory[]` itself
(beyond the 2.2.2 boundary-condition fix) — the raw-address-read technique this research validates
would keep working across all of them. The only real bump-risk this project would inherit is the
`v2.4.0` frame-timing note interacting with `navigation.py`'s existing hold-until-position-changes logic.

## Answer to the ticket's three questions

1. **Yes** — `wOverworldMap` (`0xC6E8`–`0xCBFC`, 1300 bytes) holds the complete current map's block-ID
   layout, addressed by the map's real `wCurMapWidth`/`wCurMapHeight` (not screen dimensions), populated
   fresh on every map load by `LoadTileBlockMap`. It is block-ID data, not a ready-made collision bitmap.
2. **Yes, already reachable at the pinned PyBoy 2.2.0**, via a plain `pyboy.memory[0xC6E8:0xCBFC]` read
   — the same generic, unrestricted low-level accessor this repo's `game_state.py` already uses for
   every other field, not gated behind any higher-level wrapper API. No version bump is required for
   raw reachability.
3. **A version bump is not required, and would not by itself close the gap either**: even PyBoy's
   current `master` still only derives collision from the on-screen window, identically to 2.2.0. The
   block→tile→walkable-list decode needed to turn `wOverworldMap` into a usable full-map collision grid
   is work this project would have to do itself at any PyBoy version. The one concrete reason to care
   about a version at all is `v2.4.0`'s documented frame-timing correction, which would need
   `navigation.py`'s existing timing-sensitive button-hold logic re-verified against a real boot before
   relying on it post-bump.

## Open follow-ups for a later session

- This pass is disassembly/source reading only, not a build/verify pass: the local `uv run` environment
  in this worktree failed to build `pyboy==2.2.0` from source (a Cython compile error in
  `pyboy/core/cartridge/cartridge.py`) when this research was done, so `wOverworldMap`'s contents were
  not independently re-verified by booting this repo's own `pokemon_red.gb` and reading `0xC6E8` live,
  the way this repo's other RAM addresses were (per `game_state.py`'s own docstring convention). Before
  any implementation lands, re-run this project's normal `uv run pytest` (once the build issue is fixed
  or a prebuilt wheel is available) and confirm `pyboy.memory[0xC6E8:0xCBFC]` against a known map's
  actual block layout, the same way this repo's other addresses were boot-verified.
- If #84/#85 move forward with building the block→tile→walkable expansion, the exact byte layout of
  `wTilesetBlocksPtr`'s ROM blockset data (how a block ID maps to its 16 tile IDs) still needs to be
  pulled from `pret/pokered`'s tileset-loading code and cited the same way this document cited
  `LoadTileBlockMap` — not attempted here since it was out of this ticket's scope (WRAM reachability
  only, not a full implementation design).
