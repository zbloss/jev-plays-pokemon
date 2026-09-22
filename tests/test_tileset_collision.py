from collections import deque
from pathlib import Path

import pytest
from pyboy import PyBoy

from jev_plays_pokemon import rom_maps
from jev_plays_pokemon.tileset_collision import is_walkable, parse_tileset_headers

ROM_PATH = Path(__file__).resolve().parent.parent / "pokemon_red.gb"

# `pokemon_red.gb` is gitignored - same skip pattern as
# tests/test_navigation.py/tests/test_rom_maps.py.
pytestmark = pytest.mark.skipif(
    not ROM_PATH.exists(), reason=f"{ROM_PATH} not present locally"
)

_TILESET_OVERWORLD = 0
_TILESET_FOREST = 3

# `pret/pokered`'s `data/tilesets/collision_tile_ids.asm` - `Overworld_Coll`/
# `Forest_Coll`'s own listed tile IDs, hand-transcribed from that file at
# the same pinned commit `docs/research/gen1-map-coordinate-sources.md`
# uses - independent of this module's own ROM-derived decode, so equality
# below is a real cross-check, not a tautology.
_OVERWORLD_COLL_TILE_IDS = frozenset(
    {
        0x00,
        0x10,
        0x1B,
        0x20,
        0x21,
        0x23,
        0x2C,
        0x2D,
        0x2E,
        0x30,
        0x31,
        0x33,
        0x39,
        0x3C,
        0x3E,
        0x52,
        0x54,
        0x58,
        0x5B,
    }
)
_FOREST_COLL_TILE_IDS = frozenset(
    {
        0x1E,
        0x20,
        0x2E,
        0x30,
        0x34,
        0x37,
        0x39,
        0x3A,
        0x40,
        0x51,
        0x52,
        0x5A,
        0x5C,
        0x5E,
        0x5F,
    }
)

# Directly, uniquely located in this repo's own `pokemon_red.gb` (see
# `tileset_collision.py`'s module docstring): the Overworld block-set's
# 2048-byte body matches `gfx/blocksets/overworld.bst` byte-for-byte at
# exactly this file offset, and nowhere else.
_OVERWORLD_BLOCK_FILE_OFFSET = 411104

_MAP_ROUTE_3 = 14
_PEWTER_TO_ROUTE3_STATE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "pewter_to_route3.state"
)


@pytest.fixture(scope="module")
def rom() -> bytes:
    return rom_maps.load_rom(ROM_PATH)


@pytest.fixture(scope="module")
def headers(rom: bytes):
    return parse_tileset_headers(rom)


def test_overworld_header_matches_directly_verified_rom_offsets(headers):
    header = headers[_TILESET_OVERWORLD]
    assert header.block_file_offset == _OVERWORLD_BLOCK_FILE_OFFSET
    assert header.passable_tile_ids == _OVERWORLD_COLL_TILE_IDS


def test_forest_passable_tile_ids_match_pret_pokered(headers):
    """A second tileset, independent of the one entry
    `tileset_collision.py`'s anchor byte-pattern search targeted - confirms
    the table's stride/order, not just its start."""
    assert headers[_TILESET_FOREST].passable_tile_ids == _FOREST_COLL_TILE_IDS


def test_is_walkable_returns_none_off_map(rom, headers):
    rmap = rom_maps.parse_map(rom, _MAP_ROUTE_3)
    assert is_walkable(rom, rmap, headers, -1, 0) is None
    assert is_walkable(rom, rmap, headers, 0, -1) is None
    width_tiles = rmap.width_blocks * 2
    height_tiles = rmap.height_blocks * 2
    assert is_walkable(rom, rmap, headers, width_tiles, 0) is None
    assert is_walkable(rom, rmap, headers, 0, height_tiles) is None


def test_is_walkable_matches_pyboys_live_collision_map_at_the_route3_landing_tile(
    rom, headers
):
    """Cross-checks this module's static decode against PyBoy's own live,
    dynamic `game_area_collision()` - a different code path entirely (see
    `navigation.py`'s docstring) - over every on-screen tile at a real,
    booted position. Zero mismatches is the module docstring's own claim
    for why the raw-tile-quadrant sampling (far corner, not near) is right,
    not just plausible.
    """
    rmap = rom_maps.parse_map(rom, _MAP_ROUTE_3)
    pyboy = PyBoy(str(ROM_PATH), window="null")
    pyboy.set_emulation_speed(0)
    try:
        with _PEWTER_TO_ROUTE3_STATE_PATH.open("rb") as f:
            pyboy.load_state(f)
        pyboy.tick(1, False)
        memory = pyboy.memory
        player_x, player_y = memory[0xD362], memory[0xD361]
        assert memory[0xD35E] == _MAP_ROUTE_3

        collision = pyboy.game_area_collision()
        compared = 0
        for block_row in range(9):
            for block_col in range(10):
                world_x = player_x + (block_col - 4)
                world_y = player_y + (block_row - 4)
                expected = is_walkable(rom, rmap, headers, world_x, world_y)
                if expected is None:
                    continue
                compared += 1
                live = bool(collision[block_row * 2, block_col * 2])
                assert live == expected, (world_x, world_y, live, expected)
        assert compared > 0
    finally:
        pyboy.stop(save=False)


def test_route3_has_a_static_walkable_path_from_the_pewter_crossing_to_the_route4_connection(
    rom, headers
):
    """#113: the live-emulator BFS in PR #115 found a bounded dead end
    walking east from Pewter City's own Route 3 crossing (world tile
    `(0, 8)`), capped at world x=22 across every row tried in the y=4-13
    band. This is the static, whole-map counterpart to that same search -
    and it finds a real, ordinary-walkable path all the way to Route 3's
    own north connection to Route 4 (`travel_graph.py`'s own computed hop,
    world tile `(59, 0)`) - the previous BFS's dead end wasn't real, it
    just didn't explore far enough north to find the way around.
    """
    rmap = rom_maps.parse_map(rom, _MAP_ROUTE_3)
    start = (0, 8)
    target = (59, 0)

    seen = {start}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor = (x + dx, y + dy)
            if neighbor in seen:
                continue
            if is_walkable(rom, rmap, headers, *neighbor):
                seen.add(neighbor)
                queue.append(neighbor)

    assert target in seen
