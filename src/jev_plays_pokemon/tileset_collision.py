"""Static per-tile walkability, decoded directly from the ROM's own
tileset collision + block data - the ROM-side counterpart to `rom_maps.py`'s
map geometry.

#113's own follow-up names the gap this fills: PyBoy 2.2.0's Gen1 wrapper
only exposes a *live* collision map for whatever's currently on screen
(`game_area_collision()`, scrolled to follow the player - see
`navigation.py`'s own docstring), so answering "is there a walkable route
between these two tiles" for anything wider than one screen has, until now,
needed either walking it for real or an exhaustive save-state-forked BFS
(both tried against Route 3's own boulder formation past Pewter City,
per #113's history). This module answers the same question directly from
`pokemon_red.gb`'s own bytes instead - no PyBoy boot required - the same
"read the artifact, not the docs" discipline `rom_maps.py` already applies
to map headers, applied here to `pret/pokered`'s tileset headers.

## Tileset header table

`pret/pokered`'s `data/tilesets/tileset_headers.asm` (`Tilesets:`, one
`tileset` macro invocation per `tileset_constants.asm` id, `NUM_TILESETS`
(24) of them) compiles to a fixed-size table: one 12-byte entry per
tileset -

```text
+0  db  bank (of the block/gfx pointers below)
+1  dw  block-set pointer   (block ID -> 16 raw tile IDs, 4x4, row-major)
+3  dw  gfx pointer          (not used here)
+5  dw  collision pointer    (list of passable raw tile IDs, $FF-terminated)
+7  db  counter tile 1
+8  db  counter tile 2
+9  db  counter tile 3
+10 db  grass tile
+11 db  animation kind
```

`_TILESET_HEADER_TABLE_FILE_OFFSET` was found by anchoring, not guessed:
`pret/pokered` (pinned at the same commit `docs/research/gen1-map-
coordinate-sources.md` uses, see that doc) gives the Overworld tileset's
own block-set (`gfx/blocksets/overworld.bst`, 2048 bytes) and collision
list (`data/tilesets/collision_tile_ids.asm`'s `Overworld_Coll`) byte-for-
byte - both found as a single, unique match in this repo's own
`pokemon_red.gb` (2048-byte block-set at file offset 411104 / bank 25;
20-byte, `$FF`-terminated collision list at file offset 5941 / bank 0).
Resolving those two file offsets back to (bank, CPU address) pairs and
searching for the resulting `[bank][block_lo][block_hi][??][??][coll_lo]
[coll_hi]` byte pattern (wildcarding the two gfx-pointer bytes, which
weren't independently known) finds exactly one match - at file offset
51134 - which this module takes as entry 0 (`OVERWORLD`) of the table.
Decoding all 24 entries at a 12-byte stride from there reproduces every
tileset's counter/grass/animation bytes byte-for-byte against `pret/
pokered`'s own source (e.g. entry 3, `FOREST`: counters `-1,-1,-1`, grass
`$20`, animation `TILEANIM_WATER` (`01`) - exactly `tileset Forest, -1,
-1, -1, $20, TILEANIM_WATER`), independent confirmation the anchor and
stride are both right, not just the one entry searched for.

## Pointer resolution: banked vs fixed

Every tileset's collision pointer decodes below `$4000` (Game Boy CPU
address space's fixed, always-mapped bank-0 window), while every
block-set pointer decodes at or above it (the switchable `$4000`-`$7FFF`
window, needing the entry's own bank byte) - confirmed across all 24
entries, not assumed from one. `_resolve_pointer_file_offset` branches on
exactly that: a fixed-window address's file offset *is* the address
(bank-0 CPU addresses and file offsets already coincide - `rom_maps.py`'s
own `_file_offset` only ever handles the banked case, since every pointer
it resolves happens to be banked).

## World tile -> raw tile

A map's own block grid (`rom_maps.RomMap.blocks`) is already in the same
"2x2 tiles per block" frame `rom_maps.py`'s docstring documents - one
block per `(y // 2, x // 2)` - each block's 16 raw tile IDs covering a
2x2 arrangement of those same world tiles (4x4 raw tiles). Each world tile
is therefore itself a 2x2 group of raw tiles, and `pret/pokered`'s own
collision check (`home/overworld.asm`'s `CheckTilePassable`) tests exactly
one of those four: the ID it tests comes from
`engine/overworld/player_state.asm`'s `_GetTileAndCoordsInFrontOfPlayer`,
which reads `lda_coord 8, 11` / `8, 7` / `6, 9` / `10, 9` for
down/up/left/right, and `macros/coords.asm`'s `coord` macro expands
`lda_coord x, y` to `(y) * SCREEN_WIDTH + (x) + wTileMap` - so every
direction samples the *same* sub-tile of the destination cell, two 8x8
units per world-tile step, and which sub-tile that constant `(8, 9)` offset
picks is fixed, not a free parameter.

Which one it picks is settled by measurement rather than by reading alone.
Booting a real map and scanning WRAM locates `wTileMap` itself at `$C3A0` -
its 360 bytes are byte-identical to vBGMap0's visible 20x18 - and fitting
the world grid against that buffer puts the player's own 16x16 cell at
tilemap columns 8-9 x rows 8-9 (0 mismatches over all 81 visible cells on
Mt Moon B2F, 45 on 1F). `hlcoord 8, 9` is thus the cell's *left* column and
*bottom* row, so this module samples `(sub_row 1, sub_col 0)` of the
quadrant a world tile occupies inside its block.

The choice of sub-tile is not a detail. Mt Moon's stair block is
`{(0, 0): 10, (0, 1): 11, (1, 0): 26, (1, 1): 27}` and `CAVERN`'s passable
list contains exactly one of those four (`26`) - and the game does walk onto
Mt Moon 1F's `(17, 11)` stair and warp from it, so the tile it tests has to
be `(1, 0)`. Sampling `(1, 1)` reads `27` there and reports every stair in
the dungeon as a wall; the same wrong corner reads raw `22` (impassable)
where `(1, 0)` reads `21` (passable) at `(3, 5)` on Mt Moon B2F - the single
tile joining the `(5, 7)` stair's pocket to the open floor north of it -
which reported the whole northwest wing, and with it the only way out of the
dungeon towards Route 4, as sealed. PyBoy's live `game_area_collision()`
cannot arbitrate between the corners: it too reports one fixed sub-tile per
cell (which is how `tests/test_tileset_collision.py` reads it, at even
indices), and on the blocks where the two readings disagree both left-column
tiles are `21`, so its answer tracks this one rather than the old one.

## What this doesn't model

Three Gen 1 movement mechanics sit outside plain per-tile passability and
aren't modeled here: ledges (a handful of raw tile IDs, listed in
`pret/pokered`'s `data/tilesets/ledge_tiles.asm`, that read as impassable
under an ordinary check but are a one-directional hop from a specific
standing tile); tile-pair collisions (`pair_collision_tile_ids.asm`, a
same-tileset blocked-pair list for elevation changes - `CAVERN` and `FOREST`
entries and no `OVERWORLD` one, which is the only tileset this ticket's own
Route 3 problem needed); and gym spinners, whose tile IDs
`data/tilesets/spinner_tiles.asm` lists as ordinary tile IDs that appear in
their own tileset's passable list like any floor tile, and which
`engine/overworld/spinners.asm` then uses to throw the player across the room.
The first and third are both "this module says walkable, and the game does
something other than walk" cases, which is why `raw_tile_id` is public: a
caller planning around them needs the ID, not just the yes/no. A route this
module finds is real ordinary walking; a route it doesn't find might still
exist via a ledge jump this module can't see.
"""

from __future__ import annotations

from dataclasses import dataclass

from jev_plays_pokemon.rom_maps import RomMap

_TILESET_HEADER_TABLE_FILE_OFFSET = 51134
_TILESET_HEADER_ENTRY_SIZE = 12
_TILESET_COUNT = 24
_ROM_BANK_SIZE = 0x4000
_BANKED_CPU_BASE = 0x4000
_BLOCK_TILE_COUNT = 16  # 4x4 raw tiles per block
_BLOCK_WIDTH = 4
# `lda_coord 8, 9` inside a world tile's own 2x2 raw group: column 8 is the
# group's left column, row 9 its bottom row (see the module docstring).
_SAMPLE_ROW_IN_CELL = 1
_SAMPLE_COL_IN_CELL = 0
_COLLISION_LIST_TERMINATOR = 0xFF


@dataclass(frozen=True)
class TilesetHeader:
    """One tileset's decoded block-set/collision pointers, per the module
    docstring's table layout."""

    tileset_id: int
    block_file_offset: int
    passable_tile_ids: frozenset[int]


def _resolve_pointer_file_offset(bank: int, address: int) -> int:
    """A tileset header pointer's file offset - see the module docstring's
    "Pointer resolution" section for why this isn't always banked."""
    if address < _BANKED_CPU_BASE:
        return address
    return bank * _ROM_BANK_SIZE + (address - _BANKED_CPU_BASE)


def _parse_passable_tile_ids(rom: bytes, coll_file_offset: int) -> frozenset[int]:
    tile_ids = []
    offset = coll_file_offset
    while rom[offset] != _COLLISION_LIST_TERMINATOR:
        tile_ids.append(rom[offset])
        offset += 1
    return frozenset(tile_ids)


def parse_tileset_headers(rom: bytes) -> tuple[TilesetHeader, ...]:
    """Every tileset's decoded header, indexed by tileset id (matching
    `tileset_constants.asm`'s `const_def` order, same as `RomMap.
    tileset_id`) - see the module docstring for the table's anchor/layout."""
    headers = []
    for tileset_id in range(_TILESET_COUNT):
        entry_offset = (
            _TILESET_HEADER_TABLE_FILE_OFFSET + tileset_id * _TILESET_HEADER_ENTRY_SIZE
        )
        bank = rom[entry_offset]
        block_address = rom[entry_offset + 1] | (rom[entry_offset + 2] << 8)
        coll_address = rom[entry_offset + 5] | (rom[entry_offset + 6] << 8)
        headers.append(
            TilesetHeader(
                tileset_id=tileset_id,
                block_file_offset=_resolve_pointer_file_offset(bank, block_address),
                passable_tile_ids=_parse_passable_tile_ids(
                    rom, _resolve_pointer_file_offset(bank, coll_address)
                ),
            )
        )
    return tuple(headers)


def _raw_tile_id(
    rom: bytes, block_file_offset: int, block_id: int, x: int, y: int
) -> int:
    """The specific raw tile `pret/pokered`'s own collision check samples
    for world-tile quadrant `(x % 2, y % 2)` - see the module docstring's
    "World tile -> raw tile" section for why that is the quadrant's bottom-left
    tile and not one of the other three."""
    block_offset = block_file_offset + block_id * _BLOCK_TILE_COUNT
    raw_row = (y % 2) * 2 + _SAMPLE_ROW_IN_CELL
    raw_col = (x % 2) * 2 + _SAMPLE_COL_IN_CELL
    return rom[block_offset + raw_row * _BLOCK_WIDTH + raw_col]


def raw_tile_id(
    rom: bytes, rmap: RomMap, headers: tuple[TilesetHeader, ...], x: int, y: int
) -> int | None:
    """The raw tile ID behind world tile `(x, y)` - the same one `is_walkable`
    looks up in the tileset's passable list, exposed because passability is not
    the only thing a tile's ID decides. A gym's spinner tiles are the case in
    point: `pret/pokered`'s `data/tilesets/spinner_tiles.asm` lists them as
    ordinary tile IDs, they appear in their tileset's passable list like any
    floor tile, and `engine/overworld/spinners.asm` then throws the player
    across the room from them - so a caller planning a route needs the ID, not
    just the yes/no. `None` if `(x, y)` isn't on the map at all."""
    width_tiles = rmap.width_blocks * 2
    height_tiles = rmap.height_blocks * 2
    if not (0 <= x < width_tiles and 0 <= y < height_tiles):
        return None
    block_col, block_row = x // 2, y // 2
    block_id = rmap.blocks[block_row * rmap.width_blocks + block_col]
    header = headers[rmap.tileset_id]
    return _raw_tile_id(rom, header.block_file_offset, block_id, x, y)


def is_walkable(
    rom: bytes, rmap: RomMap, headers: tuple[TilesetHeader, ...], x: int, y: int
) -> bool | None:
    """Whether world tile `(x, y)` on `rmap` is ordinary-walkable (see the
    module docstring's "What this doesn't model" section for what "ordinary"
    excludes) - `None` if `(x, y)` isn't on the map at all."""
    tile = raw_tile_id(rom, rmap, headers, x, y)
    if tile is None:
        return None
    return tile in headers[rmap.tileset_id].passable_tile_ids
