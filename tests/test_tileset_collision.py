from collections import deque
from pathlib import Path

import pytest
from pyboy import PyBoy

from jev_plays_pokemon import rom_maps
from jev_plays_pokemon.tileset_collision import (
    crosses_blocked_pair,
    is_walkable,
    parse_tile_pair_collisions,
    parse_tileset_headers,
    raw_tile_id,
)

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


@pytest.fixture(scope="module")
def pairs(rom: bytes):
    return parse_tile_pair_collisions(rom)


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


_MAP_MT_MOON_1F = 59
_MAP_MT_MOON_B1F = 60
_MAP_MT_MOON_B2F = 61
_TILESET_CAVERN = 17

# `pret/pokered`'s `data/tilesets/collision_tile_ids.asm`'s `Cavern_Coll`,
# hand-transcribed from that file the same way the two above were.
_CAVERN_COLL_TILE_IDS = frozenset(
    {0x05, 0x15, 0x18, 0x1A, 0x20, 0x21, 0x22, 0x2A, 0x2D, 0x30}
)


def _legally_reachable(
    rom: bytes,
    rmap,
    headers,
    pair_collisions: frozenset[tuple[int, int, int]],
    start: tuple[int, int],
    opened: frozenset[tuple[int, int]] = frozenset(),
) -> frozenset[tuple[int, int]]:
    """Every tile ordinary walking reaches from `start`, honouring *both* of
    `pret/pokered`'s `home/overworld.asm` step checks plus the map's object
    records. `opened` names objects the walk is allowed to treat as already
    picked up, which is how a fossil becomes a doorway."""
    width, height = rmap.width_blocks * 2, rmap.height_blocks * 2
    walls = {(o.x, o.y) for o in rmap.objects} - opened
    seen = {start}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor = (x + dx, y + dy)
            if not (0 <= neighbor[0] < width and 0 <= neighbor[1] < height):
                continue
            if neighbor in seen or neighbor in walls:
                continue
            if is_walkable(rom, rmap, headers, *neighbor) is not True:
                continue
            if crosses_blocked_pair(
                rom, rmap, headers, pair_collisions, (x, y), neighbor
            ):
                continue
            seen.add(neighbor)
            queue.append(neighbor)
    return frozenset(seen)


def _raw_tile_quadrant(rom: bytes, rmap, headers, x: int, y: int) -> dict:
    """The 2x2 raw tile IDs world tile `(x, y)` occupies inside its 4x4 block,
    keyed by `(sub_row, sub_col)` - read straight off the block-set bytes
    `_raw_tile_id` indexes, so the tests below can name the sub-tile the decode
    picks without going through it."""
    block_id = rmap.blocks[(y // 2) * rmap.width_blocks + (x // 2)]
    base = headers[rmap.tileset_id].block_file_offset + block_id * 16
    row0, col0 = (y % 2) * 2, (x % 2) * 2
    return {
        (sub_row, sub_col): rom[base + (row0 + sub_row) * 4 + (col0 + sub_col)]
        for sub_row in (0, 1)
        for sub_col in (0, 1)
    }


def test_cavern_passable_tile_ids_match_pret_pokered(headers):
    assert headers[_TILESET_CAVERN].passable_tile_ids == _CAVERN_COLL_TILE_IDS


def test_mt_moon_stairs_are_walkable_under_the_bottom_left_sample(rom, headers):
    """Pins which of a world tile's four raw tiles the decode tests.

    Mt Moon is the discriminating map: 15 of its 17 warp tiles sit on a block
    whose 2x2 raw group is *not* uniformly passable - the down-stairs block is
    `{(0, 0): 8, (0, 1): 9, (1, 0): 24, (1, 1): 25}`, the up-stairs block
    `{(0, 0): 10, (0, 1): 11, (1, 0): 26, (1, 1): 27}` - and `CAVERN`'s passable
    list contains exactly one tile of each four (`24` and `26`), always the
    `(1, 0)` one and never the `(1, 1)` one this module used to sample. (The two
    exceptions are `(14, 35)`/`(15, 35)`, the Route 3 entrance tiles, uniform
    `33` in all four.) A warp the player reaches by walking has to be a tile
    `CheckTilePassable` accepts - both of `home/overworld.asm`'s warp checks,
    `CheckWarpsCollision` and `CheckWarpsNoCollision`, compare the tile the
    player stands *on*, so a stair that failed the passability check could never
    be entered at all - which makes the passable sub-tile the sampled sub-tile,
    asserted here for fifteen independently-placed tiles at once rather than for
    one.
    """
    for map_id in (_MAP_MT_MOON_1F, _MAP_MT_MOON_B1F, _MAP_MT_MOON_B2F):
        rmap = rom_maps.parse_map(rom, map_id)
        passable = headers[rmap.tileset_id].passable_tile_ids
        discriminating = 0
        for warp in rmap.warps:
            quadrant = _raw_tile_quadrant(rom, rmap, headers, warp.x, warp.y)
            walkable = is_walkable(rom, rmap, headers, warp.x, warp.y)
            assert walkable is True, (map_id, warp.x, warp.y, quadrant)
            assert raw_tile_id(rom, rmap, headers, warp.x, warp.y) in passable
            open_sub_tiles = {sub for sub, tile in quadrant.items() if tile in passable}
            if open_sub_tiles != {(0, 0), (0, 1), (1, 0), (1, 1)}:
                discriminating += 1
                assert open_sub_tiles == {(1, 0)}, (map_id, warp.x, warp.y, quadrant)
        assert discriminating > 0


def test_mt_moon_b2f_route_to_the_route4_stair_needs_a_fossil_and_the_other_stair(
    rom, headers, pairs
):
    """#99/#100's route out of Mt Moon, in static form - and the measurement
    that overturned this file's earlier version of the same claim.

    `(5, 7)` is the stair that lands in B1F's Route 4 chamber, the only way to
    Route 4's east side and so the only land route to Cerulean City and
    everything beyond it. This file used to assert it was reachable from
    `(25, 9)` on `is_walkable` alone. That is wrong, and Mt Moon is exactly
    where `crosses_blocked_pair` earns its place: the flood from `(25, 9)`
    covers 483 tiles without the pair rule and **67** with it, because
    `(25, 9)`'s landing band is `$20` and every corridor leaving it is `$05` -
    a `CAVERN $20 <-> $05` crossing the ROM refuses in both directions. That
    landing is a sealed pocket, and its only warp is the stair it came in by.

    The floor's real trunk road starts at `(21, 17)` (B1F's `(21, 17)` stair,
    384 tiles with the pair rule) and runs to the two fossil pedestals at
    `(12, 6)`/`(13, 6)`, which `rom_maps` reports as sprite 62 objects sitting
    on walkable floor. Taking either fossil is what opens the dungeon: with one
    gone the component grows to 448 tiles and includes `(5, 7)`. So Mt Moon's
    exit is not merely a walking problem - the ROM's own collision data makes
    the fossil pickup a prerequisite for leaving, which is why the walk that
    stalls here has nothing to do with how it plans.
    """
    rmap = rom_maps.parse_map(rom, _MAP_MT_MOON_B2F)
    target = (5, 7)
    assert is_walkable(rom, rmap, headers, *target) is True
    assert is_walkable(rom, rmap, headers, 3, 5) is True

    sealed_pocket = _legally_reachable(rom, rmap, headers, pairs, (25, 9))
    assert target not in sealed_pocket
    assert len(sealed_pocket) == 67

    without_fossil = _legally_reachable(rom, rmap, headers, pairs, (21, 17))
    assert target not in without_fossil
    assert len(without_fossil) == 384

    with_fossil = _legally_reachable(
        rom, rmap, headers, pairs, (21, 17), opened=frozenset({(12, 6)})
    )
    assert target in with_fossil
    assert len(with_fossil) == 448


def _legally_reachable(
    rom: bytes,
    rmap,
    headers,
    pair_collisions: frozenset[tuple[int, int, int]],
    start: tuple[int, int],
    opened: frozenset[tuple[int, int]] = frozenset(),
) -> frozenset[tuple[int, int]]:
    """Every tile ordinary walking reaches from `start`, honouring *both* of
    `pret/pokered`'s `home/overworld.asm` step checks plus the map's object
    records. `opened` names objects the walk is allowed to treat as already
    picked up, which is how a fossil becomes a doorway."""
    width, height = rmap.width_blocks * 2, rmap.height_blocks * 2
    walls = {(o.x, o.y) for o in rmap.objects} - opened
    seen = {start}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor = (x + dx, y + dy)
            if not (0 <= neighbor[0] < width and 0 <= neighbor[1] < height):
                continue
            if neighbor in seen or neighbor in walls:
                continue
            if is_walkable(rom, rmap, headers, *neighbor) is not True:
                continue
            if crosses_blocked_pair(
                rom, rmap, headers, pair_collisions, (x, y), neighbor
            ):
                continue
            seen.add(neighbor)
            queue.append(neighbor)
    return frozenset(seen)


# `pret/pokered`'s `data/tilesets/pair_collision_tile_ids.asm`, hand-transcribed
# from that file at the same pinned commit `docs/research/gen1-map-coordinate-
# sources.md` uses. Independent of this module's anchor, so the equality below
# checks both where the parse started and what it stopped at.
_TILE_PAIR_COLLISION_ROWS = frozenset(
    {
        # TilePairCollisionsLand
        (_TILESET_CAVERN, 0x20, 0x05),
        (_TILESET_CAVERN, 0x41, 0x05),
        (_TILESET_FOREST, 0x30, 0x2E),
        (_TILESET_CAVERN, 0x2A, 0x05),
        (_TILESET_CAVERN, 0x05, 0x21),
        (_TILESET_FOREST, 0x52, 0x2E),
        (_TILESET_FOREST, 0x55, 0x2E),
        (_TILESET_FOREST, 0x56, 0x2E),
        (_TILESET_FOREST, 0x20, 0x2E),
        (_TILESET_FOREST, 0x5E, 0x2E),
        (_TILESET_FOREST, 0x5F, 0x2E),
        # TilePairCollisionsWater
        (_TILESET_FOREST, 0x14, 0x2E),
        (_TILESET_FOREST, 0x48, 0x2E),
        (_TILESET_CAVERN, 0x14, 0x05),
    }
)


def test_tile_pair_collisions_match_pret_pokered_row_for_row(pairs):
    """All fourteen rows of both tables, from an offset the module found by
    anchoring six bytes - and the terminator positions are what separate the
    second table's three rows from whatever the ROM keeps after it, so a
    mis-read here shows up as a row that isn't in pokered or a row missing."""
    assert pairs == _TILE_PAIR_COLLISION_ROWS


def test_tile_pair_collisions_only_name_the_two_tilesets_that_need_them(pairs):
    """`FOREST` and `CAVERN` only. The nine `FOREST` rows all pair against
    `$2E` and the five `CAVERN` rows all pair against `$05`, which is why
    neither rule can wall off a map that never uses those floor IDs."""
    assert {row[0] for row in pairs} == {_TILESET_FOREST, _TILESET_CAVERN}
    for tileset_id, partner in ((_TILESET_FOREST, 0x2E), (_TILESET_CAVERN, 0x05)):
        rows = {row for row in pairs if row[0] == tileset_id}
        assert rows
        assert all(partner in (row[1], row[2]) for row in rows), rows


def test_the_pair_rule_is_what_seals_mt_moon_b2fs_landing_pocket(rom, headers, pairs):
    """A regression for the rule itself, at the tile that made it visible.

    `(24, 12)` is ordinary walkable - raw `$05`, squarely in `CAVERN`'s
    passable list - and it sits on a corridor the passable list calls open the
    whole width of the floor. `(24, 11)`, the tile directly above it, is raw
    `$20`. `is_walkable` calls both walkable and so plans the step; the ROM
    refuses it, and every one of the refusals this repo's walker recorded in
    this dungeon is a crossing of exactly that kind.
    """
    rmap = rom_maps.parse_map(rom, _MAP_MT_MOON_B2F)
    assert is_walkable(rom, rmap, headers, 24, 12) is True
    assert is_walkable(rom, rmap, headers, 24, 11) is True
    assert raw_tile_id(rom, rmap, headers, 24, 12) == 0x05
    assert raw_tile_id(rom, rmap, headers, 24, 11) == 0x20
    assert crosses_blocked_pair(rom, rmap, headers, pairs, (24, 11), (24, 12))
    assert crosses_blocked_pair(rom, rmap, headers, pairs, (24, 12), (24, 11))

    pocket = _legally_reachable(rom, rmap, headers, pairs, (25, 9))
    assert (24, 12) not in pocket
    assert all(
        not crosses_blocked_pair(
            rom, rmap, headers, pairs, here, (here[0] + dx, here[1] + dy)
        )
        for here in pocket
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
        if (here[0] + dx, here[1] + dy) in pocket
    )


_MAP_ROUTE_4 = 15


def _warp_landing(rmap, warp):
    """The tile a warp record actually lands the player on, read out of the
    destination map's own warp list (`rom_maps.MapWarp.dest_warp` is a 0-based
    index into it, and `dest_map=None` is `LAST_MAP` - which inside Mt Moon is
    the last *outside* map, so Route 4, per `pret/pokered`'s `WarpFound2` only
    writing `wLastMap` when leaving an outside map)."""
    return rmap.warps[warp.dest_warp]


def test_the_route_from_route4s_route3_landing_to_ceruleans_gate_is_one_walkable_chain(
    rom, headers, pairs
):
    """The whole claim #99's remaining four milestones rest on, link by link.

    Route 3's north connection puts the player on Route 4 at `(9, 17)` (the
    committed `route3_to_route4.state` fixture's landing tile), and Route 4's
    decoded walkability is "right and useless on its own": the wall at x=20-23
    splits the map, and `travel_graph.py`'s own docstring calls the only
    connector a scripted concern rather than a routing one. This walks that
    connector statically, and asserts each hop's landing tile *from the warp
    records* rather than from the coordinates the last attempt assumed - which
    is where the earlier version of this chain was wrong.

    Every link is on the ROM's own bytes, because a link that fails statically
    fails on live hardware too; each one that passes is a leg the walker can be
    pointed at, and the two that need something beyond walking - the fossil,
    and Route 4's one-way ledge drop into Cerulean's side of the map - are
    named as such instead of planned through.

    Worth recording next to the ledge: `travel_graph.connection_hop` picks
    Route 4's crossing tile for its `east -> Cerulean City` connection as
    `(89, 8)` - the midpoint of the two maps' overlapping rows - and `$14`
    there is not in `OVERWORLD`'s passable list. The crossing is real (rows 10
    and 11 of column 89 are walkable and land in Cerulean), but the graph's
    chosen representative tile is one the player cannot stand on, so a walk
    driven by that hop stalls one tile short of a map edge that is otherwise
    open.
    """
    route4 = rom_maps.parse_map(rom, _MAP_ROUTE_4)
    moon_1f = rom_maps.parse_map(rom, _MAP_MT_MOON_1F)
    b1f = rom_maps.parse_map(rom, _MAP_MT_MOON_B1F)
    b2f = rom_maps.parse_map(rom, _MAP_MT_MOON_B2F)

    # Route 4's west half reaches the Mt Moon door, and the door's record
    # really is Mt Moon 1F's.
    landing = _legally_reachable(rom, route4, headers, pairs, (9, 17))
    door = next(w for w in route4.warps if (w.x, w.y) == (18, 5))
    assert (door.x, door.y) in landing
    assert door.dest_map == _MAP_MT_MOON_1F
    inside = _warp_landing(moon_1f, door)
    assert (inside.x, inside.y) in _legally_reachable(
        rom, moon_1f, headers, pairs, (inside.x, inside.y)
    )

    # 1F's whole floor is one component, and it holds the stair down.
    from_door = _legally_reachable(rom, moon_1f, headers, pairs, (inside.x, inside.y))
    assert (5, 5) in from_door
    stair = next(w for w in moon_1f.warps if (w.x, w.y) == (5, 5))
    assert stair.dest_map == _MAP_MT_MOON_B1F
    assert (b1f.warps[stair.dest_warp].x, b1f.warps[stair.dest_warp].y) == (5, 5)

    # B1F's landing chamber holds the stair that leads to B2F's *trunk road* -
    # and not the one to the `(25, 9)` pocket, which is a dead end.
    down_stairs = _legally_reachable(rom, b1f, headers, pairs, (5, 5))
    assert (21, 17) in down_stairs
    to_b2f = next(w for w in b1f.warps if (w.x, w.y) == (21, 17))
    assert to_b2f.dest_map == _MAP_MT_MOON_B2F
    assert (b2f.warps[to_b2f.dest_warp].x, b2f.warps[to_b2f.dest_warp].y) == (21, 17)

    # B2F: the fossil is the door, and `(5, 7)` lands in B1F's Route 4 chamber.
    past_fossil = _legally_reachable(
        rom, b2f, headers, pairs, (21, 17), opened=frozenset({(12, 6)})
    )
    assert (5, 7) in past_fossil
    up = next(w for w in b2f.warps if (w.x, w.y) == (5, 7))
    assert up.dest_map == _MAP_MT_MOON_B1F
    assert (b1f.warps[up.dest_warp].x, b1f.warps[up.dest_warp].y) == (23, 3)

    # B1F's Route 4 chamber is entered only by that stair, and holds the door.
    to_route4 = _legally_reachable(rom, b1f, headers, pairs, (23, 3))
    assert (27, 3) in to_route4
    out = next(w for w in b1f.warps if (w.x, w.y) == (27, 3))
    assert out.dest_map is None, "Mt Moon's Route 4 exit is a LAST_MAP warp"
    back = route4.warps[out.dest_warp]
    assert (back.x, back.y) == (24, 5)

    # Route 4's east half opens from that landing as far as ordinary walking
    # goes: to column 79, on rows 6 and 8. What joins that frontier to the
    # map's east edge is Route 4's one-way ledge drop (down from row 8 onto row
    # 10, over a row of `$37`/`$36` drop tiles), which is the second of the two
    # mechanics `tileset_collision.py`'s docstring names as deliberately
    # unmodeled - so this test stops here rather than planning through it, and
    # the ledge is what the live walk has to actually press.
    east_of_moon = _legally_reachable(rom, route4, headers, pairs, (24, 5))
    width_tiles = route4.width_blocks * 2
    assert max(x for x, _ in east_of_moon) == 79
    assert {(79, 6), (79, 8)} <= east_of_moon
    for row in (10, 11):
        assert is_walkable(rom, route4, headers, width_tiles - 1, row) is True
        assert (width_tiles - 1, row) not in east_of_moon
