"""Parse `pokemon_red.gb` itself for per-map block/object/warp/connection data.

Formalizes the scratch parser validated in
`docs/research/gen1-map-coordinate-sources.md` (#87): every map's geometry
and event data is a byte in the ROM, addressable through the ROM's own
`MapHeaderPointers`/`MapHeaderBanks` tables - no PyBoy boot, no WRAM read,
and no per-map hardcoded offset required. This is the "read the artifact,
not the docs" discipline `game_state.py` already applies to RAM, applied
here to the ROM file instead.

`docs/adr/0002-travel-graph-for-cross-map-navigation.md`'s travel graph
needs exactly this data (warp/connection hops); this module is that data
source, with no opinion yet on graph search or milestone target selection.

## Map header layout

Each map ID resolves to a (bank, address) header location via two
fixed-size tables the ROM itself carries - `MapHeaderPointers` (a `dw` per
map ID) and `MapHeaderBanks` (a `db` per map ID) - located once by
`.blk`-anchoring (see the research doc) at ROM file offsets
`_MAP_HEADER_POINTERS_FILE_OFFSET`/`_MAP_HEADER_BANKS_FILE_OFFSET`. From
there, per `pret/pokered`'s `macros/scripts/maps.asm` (`map_header`/
`connection`/`end_map_header`), all offsets bank-relative from the header:

```text
+0  db tileset id
+1  db height (blocks)
+2  db width (blocks)
+3  dw block-grid pointer
+5  dw text pointers
+7  dw script pointer
+9  db connection mask (EAST=1 WEST=2 SOUTH=4 NORTH=8)
+10 one 11-byte connection record per set bit, in EAST,WEST,SOUTH,NORTH order
+10+11n  dw object-data pointer
```

A block is 2x2 tiles; the block grid at the header's block-grid pointer is
`height * width` bytes, one block ID per cell, row-major.

The object-data pointer leads to warp/sign/object records, each list
length-prefixed, in that fixed order (`pret/pokered`'s `object_const_defs`/
`event_const_defs` compiled layout):

```text
db border block id
db warp count      | warp count * [Y, X, dest_warp_id-1, dest_map]
db sign count       | sign count * [Y, X, text id]           (not exposed)
db object count      | object count * [sprite, Y+4, X+4, movement,
                                        facing, flags(+trainer/item bytes)]
```

Warp/object Y and X are tile coordinates in the same world frame as
`game_state.py`'s `player_x`/`player_y` (object records need `+4`
subtracted per the research doc's verified `raw[1] == y+4`/`raw[2] == x+4`
check across all 1466 object records with zero exceptions; warp records
already store plain tile coordinates). `dest_map == 255` on a warp is
`pret/pokered`'s `LAST_MAP` sentinel - "return to whichever map the player
came from," resolved by the game engine at runtime - surfaced here as
`None` rather than a literal map ID 255.

## Map count and unused slots

`lookup.maps.MAP_COUNT`/`is_unused_map` (backed by the same map ID list
`GameState.map_name` already uses) give the total map ID range and flag
`pokered`'s `UNUSED_MAP_*` placeholder slots. Those slots still resolve
through the header tables to *something*, but several consecutive ones
alias the identical (bank, address) pair and decode to implausible data
(a 240x92-block grid, a warp count over 100) - confirmed against this
repo's own ROM, not assumed. `parse_all_maps` skips them; `parse_map`
itself makes no such judgment and will parse any map ID it's given,
garbage included, since which IDs are "real" is a naming question
(`lookup.maps`), not a parsing one.

Total record counts here (see `tests/test_rom_maps.py`) come out lower
than the 1244 warps / 86 connections `docs/research/gen1-map-coordinate-
sources.md` reports: that research total was produced by a scratch script
that named map IDs by walking `pret/pokered`'s own
`map_header_pointers.asm` text, and for five IDs (the research doc's own
"unmatched" list - `SaffronCity`, `LancesRoom`, `Route16Gate1F`,
`RocketHideoutElevator`, `SilphCo2F`) that walk attached a real map's name
to what this module's ID-only parse shows is actually one of the
implausible unused slots above (confirmed byte-for-byte: e.g. the ID the
old script called "SaffronCity" decodes identically to `UNUSED_MAP_0B`
here). Every other map ID both parses agree on - 218 of them - decodes to
byte-identical warp/connection/object records, and this module's own count
excludes exactly those five inflated entries, plus the padding slots
around them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jev_plays_pokemon.emulator import DEFAULT_ROM_PATH
from jev_plays_pokemon.lookup.maps import MAP_COUNT, is_unused_map

_MAP_HEADER_POINTERS_FILE_OFFSET = 430
_MAP_HEADER_BANKS_FILE_OFFSET = 49725
_ROM_BANK_SIZE = 0x4000
_BANKED_CPU_BASE = 0x4000  # switchable ROM banks map in at CPU address $4000.

# Connection mask bit values (`constants/map_data_constants.asm`:
# EAST=1, WEST=2, SOUTH=4, NORTH=8), but the compiled connection *records*
# for whichever bits are set appear in north, south, west, east order -
# the authoring order `macros/scripts/maps.asm`'s `connection` macro
# enforces ("Connections go in order: north, south, west, east"), not
# ascending bit value. Iterating in bit-value order here silently paired
# each record with the wrong direction: Pallet Town's real-world north
# neighbor, Route 1, decoded as its "south" connection (and Route 21,
# its real south neighbor, as "north") until this order was corrected -
# see `tests/test_rom_maps.py`'s connection-direction assertions.
_CONNECTION_BITS: tuple[tuple[int, str], ...] = (
    (8, "north"),
    (4, "south"),
    (2, "west"),
    (1, "east"),
)
_CONNECTION_RECORD_SIZE = 11

# Connection record layout (`macros/scripts/maps.asm`'s `connection` macro,
# 11 bytes): +0 dest map id, +1..2 block pointer, +3..4 overworld-map
# pointer, +5 strip length, +6 dest map width (blocks), +7 y alignment,
# +8 x alignment, +9..10 window pointer. Only the fields this module
# surfaces (dest map, the two alignment bytes) are decoded; the rest are
# VRAM-scrolling bookkeeping `travel_graph.py` has no use for.
_CONNECTION_Y_ALIGNMENT_OFFSET = 7
_CONNECTION_X_ALIGNMENT_OFFSET = 8

# `constants/map_object_constants.asm`: object flag byte high bits.
_TRAINER_FLAG = 0x40
_ITEM_FLAG = 0x80

# `pret/pokered`'s `LAST_MAP` warp destination sentinel (see module
# docstring) - resolved to `None`, never treated as a literal map ID.
_LAST_MAP_RAW = 255


@dataclass(frozen=True)
class MapObject:
    """One decoded object-event record (an NPC, trainer, or item ball)."""

    sprite: int
    x: int
    y: int
    movement: int
    facing: int
    text_id: int
    trainer: tuple[int, int] | None = None
    item: int | None = None


@dataclass(frozen=True)
class MapWarp:
    """One decoded warp record - a door/staircase tile and its destination."""

    x: int
    y: int
    dest_warp: int
    # `None` iff the raw destination byte was `LAST_MAP` ($FF) - see module
    # docstring. Otherwise a literal destination map ID.
    dest_map: int | None


@dataclass(frozen=True)
class MapConnection:
    """One decoded map-edge connection - walk-off continuity to another map.

    `y_alignment`/`x_alignment` are the connection record's two signed
    alignment bytes (`home/overworld.asm`'s `CheckMapConnections`, copied
    verbatim into `wNorthConnectedMapYAlignment`/`...XAlignment` and the
    South/East/West equivalents by `CopyMapConnectionHeader`). Per that
    routine's own arithmetic - confirmed against this repo's ROM by cross-
    checking both directions of the Pallet Town <-> Route 1 and Viridian
    City <-> Route 1/Route 22 pairs, which agree with each other and with
    known Kanto geography:

    - north/south: the perpendicular axis (Y) snaps to `y_alignment`
      (a fixed tile row on the destination map - e.g. `height_tiles - 1`,
      the destination's south edge, for a north connection); the along-edge
      axis (X) carries over with `x_alignment` added (`dest_x = src_x +
      x_alignment`).
    - east/west: the perpendicular axis (X) snaps to `x_alignment`; the
      along-edge axis (Y) carries over with `y_alignment` added.
    """

    direction: str  # "east" | "west" | "south" | "north"
    dest_map: int
    y_alignment: int
    x_alignment: int


@dataclass(frozen=True)
class RomMap:
    """One map's full decoded header: geometry plus its event records."""

    map_id: int
    tileset_id: int
    height_blocks: int
    width_blocks: int
    # `height_blocks * width_blocks` bytes, one block ID per cell,
    # row-major - the same bytes `pret/pokered`'s `maps/<Map>.blk` files
    # carry and WRAM's `wOverworldMap` loads on map entry.
    blocks: bytes
    objects: tuple[MapObject, ...]
    warps: tuple[MapWarp, ...]
    connections: tuple[MapConnection, ...]


def load_rom(rom_path: str | Path = DEFAULT_ROM_PATH) -> bytes:
    """Read `rom_path`'s raw bytes for `parse_map`/`parse_all_maps`."""
    return Path(rom_path).read_bytes()


def _file_offset(bank: int, cpu_address: int) -> int:
    """A banked CPU address's offset into the ROM file.

    Bank `n`'s switchable window (CPU `$4000`-`$7FFF`) occupies file bytes
    `[n * 0x4000, (n + 1) * 0x4000)`; this is standard Game Boy ROM banking,
    not a `pokemon_red.gb`-specific fact.
    """
    return bank * _ROM_BANK_SIZE + (cpu_address - _BANKED_CPU_BASE)


def _signed_byte(value: int) -> int:
    """A ROM byte (0-255) as the Game Boy's two's-complement signed 8-bit
    value (-128 to 127) - how `CheckMapConnections` uses the alignment
    bytes in an `add` instruction."""
    return value - 256 if value >= 128 else value


def _parse_connections(
    rom: bytes, offset: int, mask: int
) -> tuple[tuple[MapConnection, ...], int]:
    """Decode the header's connection records, returning them plus the file
    offset immediately after the last one (where the object-data pointer
    sits - see module docstring)."""
    connections = []
    for bit, direction in _CONNECTION_BITS:
        if mask & bit:
            connections.append(
                MapConnection(
                    direction=direction,
                    dest_map=rom[offset],
                    y_alignment=_signed_byte(
                        rom[offset + _CONNECTION_Y_ALIGNMENT_OFFSET]
                    ),
                    x_alignment=_signed_byte(
                        rom[offset + _CONNECTION_X_ALIGNMENT_OFFSET]
                    ),
                )
            )
            offset += _CONNECTION_RECORD_SIZE
    return tuple(connections), offset


def _parse_event_records(
    rom: bytes, bank: int, object_data_cpu: int
) -> tuple[tuple[MapWarp, ...], tuple[MapObject, ...]]:
    """Decode the warp and object records at the header's object-data
    pointer (see module docstring for the length-prefixed record layout).
    Sign/text records sit between the two and are skipped, not exposed -
    out of this ticket's scope."""
    offset = _file_offset(bank, object_data_cpu) + 1  # skip border-block id

    warp_count = rom[offset]
    offset += 1
    warps = []
    for _ in range(warp_count):
        y, x, dest_warp, dest_map_raw = rom[offset : offset + 4]
        dest_map = None if dest_map_raw == _LAST_MAP_RAW else dest_map_raw
        warps.append(MapWarp(x=x, y=y, dest_warp=dest_warp, dest_map=dest_map))
        offset += 4

    sign_count = rom[offset]
    offset += 1 + 3 * sign_count

    object_count = rom[offset]
    offset += 1
    objects = []
    for _ in range(object_count):
        sprite, raw_y, raw_x, movement, facing, flags = rom[offset : offset + 6]
        offset += 6
        trainer: tuple[int, int] | None = None
        item: int | None = None
        if flags & _TRAINER_FLAG:
            text_id = flags & ~_TRAINER_FLAG
            trainer = (rom[offset], rom[offset + 1])
            offset += 2
        elif flags & _ITEM_FLAG:
            text_id = flags & ~_ITEM_FLAG
            item = rom[offset]
            offset += 1
        else:
            text_id = flags
        objects.append(
            MapObject(
                sprite=sprite,
                x=raw_x - 4,
                y=raw_y - 4,
                movement=movement,
                facing=facing,
                text_id=text_id,
                trainer=trainer,
                item=item,
            )
        )

    return tuple(warps), tuple(objects)


def parse_map(rom: bytes, map_id: int) -> RomMap:
    """Resolve and decode `map_id`'s full header via the ROM's own
    `MapHeaderPointers`/`MapHeaderBanks` tables - no per-map offset is
    hardcoded anywhere in this function."""
    pointer_offset = _MAP_HEADER_POINTERS_FILE_OFFSET + 2 * map_id
    header_cpu = rom[pointer_offset] | (rom[pointer_offset + 1] << 8)
    bank = rom[_MAP_HEADER_BANKS_FILE_OFFSET + map_id]
    header_offset = _file_offset(bank, header_cpu)

    tileset_id = rom[header_offset]
    height_blocks = rom[header_offset + 1]
    width_blocks = rom[header_offset + 2]
    blocks_cpu = rom[header_offset + 3] | (rom[header_offset + 4] << 8)
    connection_mask = rom[header_offset + 9]

    connections, after_connections = _parse_connections(
        rom, header_offset + 10, connection_mask
    )
    object_data_cpu = rom[after_connections] | (rom[after_connections + 1] << 8)
    warps, objects = _parse_event_records(rom, bank, object_data_cpu)

    blocks_offset = _file_offset(bank, blocks_cpu)
    block_count = height_blocks * width_blocks
    blocks = rom[blocks_offset : blocks_offset + block_count]

    return RomMap(
        map_id=map_id,
        tileset_id=tileset_id,
        height_blocks=height_blocks,
        width_blocks=width_blocks,
        blocks=blocks,
        objects=objects,
        warps=warps,
        connections=connections,
    )


def parse_all_maps(rom: bytes) -> dict[int, RomMap]:
    """Every real map ID's `RomMap`, keyed by map ID.

    Skips `lookup.maps.is_unused_map` slots (see module docstring) - callers
    needing a specific unused ID anyway can still call `parse_map` directly.
    """
    return {
        map_id: parse_map(rom, map_id)
        for map_id in range(MAP_COUNT)
        if not is_unused_map(map_id)
    }
