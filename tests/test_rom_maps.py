from pathlib import Path

import pytest

from jev_plays_pokemon import milestones, rom_maps
from jev_plays_pokemon.lookup.maps import MAP_COUNT, is_unused_map
from jev_plays_pokemon.milestone_targets import resolve_target_coordinates

ROM_PATH = Path(__file__).resolve().parent.parent / "pokemon_red.gb"

# `pokemon_red.gb` is gitignored (see test_game_state.py) - pure data-in/
# data-out against the ROM file, same skip pattern as
# tests/test_navigation.py/tests/test_game_state.py.
pytestmark = pytest.mark.skipif(
    not ROM_PATH.exists(), reason=f"{ROM_PATH} not present locally"
)

# Real, non-placeholder map IDs, per `lookup.maps`' `pokered`-derived
# constant list (see that module's `is_unused_map`).
_REAL_MAP_COUNT = sum(1 for map_id in range(MAP_COUNT) if not is_unused_map(map_id))

# Totals over that map set, counted directly against this repo's own
# `pokemon_red.gb` while building this parser. Lower than the 1244 warps /
# 86 connections `docs/research/gen1-map-coordinate-sources.md` reports -
# see `rom_maps.py`'s module docstring for why: that figure double-counted
# five IDs whose real map name got attached, by the research's own scratch
# script, to what this parser shows are unused padding slots that alias
# other maps' pointers and decode to implausible data (hundreds of warps,
# a 240x92-block grid). Every map ID both parses agree is real decodes to
# byte-identical warp/object records - checked exhaustively while building
# this module, not just for the milestone maps below.
_TOTAL_WARP_RECORDS = 813
_TOTAL_CONNECTION_RECORDS = 78

# The 14 milestone maps' NPC/item targets from
# docs/research/gen1-map-coordinate-sources.md section 2 - ROM-verified
# there (all `blocks_match_blk_file`/`records_match_asm`), in the same
# world-tile frame as `GameState.player_x`/`player_y`.
_MILESTONE_OBJECT_TARGETS: tuple[tuple[str, int, tuple[int, int]], ...] = (
    ("got_starter (Squirtle ball)", 40, (7, 3)),
    ("got_starter (Bulbasaur ball)", 40, (8, 3)),
    ("got_starter (Charmander ball)", 40, (6, 3)),
    ("got_oaks_parcel", 42, (3, 3)),
    ("got_pokedex (rival)", 40, (4, 3)),
    ("boulder_badge (Brock)", 54, (4, 1)),
    ("cascade_badge (Misty)", 65, (4, 2)),
    ("got_ss_ticket (Bill)", 88, (4, 4)),
    ("thunder_badge (Lt. Surge)", 92, (5, 1)),
    ("rainbow_badge (Erika)", 134, (4, 3)),
    ("got_poke_flute (Mr. Fuji)", 148, (10, 3)),
    ("soul_badge (Koga)", 157, (4, 10)),
    ("marsh_badge (Sabrina)", 178, (9, 8)),
    ("volcano_badge (Blaine)", 166, (3, 3)),
    ("earth_badge (Giovanni)", 45, (2, 1)),
    ("beat_champion (rival)", 120, (4, 2)),
)

# height_blocks/width_blocks/tileset_id for the same 14 milestone maps,
# read via this parser. Pallet Town/Oak's Lab/Pewter Gym's dimensions cross-
# check directly against the research doc's own worked examples (10x9,
# 6x5, 5x7 blocks respectively); the rest share the identical header-
# parsing code path, exercised and verified above via their object records.
_MILESTONE_MAP_DIMS: dict[int, tuple[int, int, int]] = {
    # map_id: (tileset_id, height_blocks, width_blocks)
    40: (5, 6, 5),  # OaksLab
    42: (2, 4, 4),  # ViridianMart
    54: (7, 7, 5),  # PewterGym
    65: (7, 7, 5),  # CeruleanGym
    88: (16, 4, 4),  # BillsHouse
    92: (7, 9, 5),  # VermilionGym
    134: (7, 9, 5),  # CeladonGym
    148: (15, 9, 10),  # PokemonTower7F
    157: (7, 9, 5),  # FuchsiaGym
    178: (22, 9, 10),  # SaffronGym
    166: (22, 9, 10),  # CinnabarGym
    45: (7, 9, 10),  # ViridianGym
    120: (7, 4, 4),  # ChampionsRoom
}


@pytest.fixture(scope="module")
def rom() -> bytes:
    return rom_maps.load_rom(ROM_PATH)


@pytest.fixture(scope="module")
def all_maps(rom: bytes) -> dict[int, rom_maps.RomMap]:
    return rom_maps.parse_all_maps(rom)


def test_map_count_matches_lookups_non_unused_map_ids(all_maps):
    assert len(all_maps) == _REAL_MAP_COUNT
    assert len(all_maps) == 226


def test_total_warp_record_count(all_maps):
    total = sum(len(m.warps) for m in all_maps.values())
    assert total == _TOTAL_WARP_RECORDS


def test_total_connection_record_count(all_maps):
    total = sum(len(m.connections) for m in all_maps.values())
    assert total == _TOTAL_CONNECTION_RECORDS


def test_unused_map_slots_are_excluded(all_maps):
    for map_id in range(MAP_COUNT):
        if is_unused_map(map_id):
            assert map_id not in all_maps


@pytest.mark.parametrize(
    "label, map_id, expected_xy",
    _MILESTONE_OBJECT_TARGETS,
    ids=[t[0] for t in _MILESTONE_OBJECT_TARGETS],
)
def test_milestone_map_has_object_at_verified_tile(rom, label, map_id, expected_xy):
    parsed = rom_maps.parse_map(rom, map_id)
    tiles = {(obj.x, obj.y) for obj in parsed.objects}
    assert expected_xy in tiles


@pytest.mark.parametrize(
    "map_id",
    sorted(_MILESTONE_MAP_DIMS),
    ids=[str(m) for m in sorted(_MILESTONE_MAP_DIMS)],
)
def test_milestone_map_dimensions(rom, map_id):
    parsed = rom_maps.parse_map(rom, map_id)
    expected_tileset, expected_h, expected_w = _MILESTONE_MAP_DIMS[map_id]
    assert (parsed.tileset_id, parsed.height_blocks, parsed.width_blocks) == (
        expected_tileset,
        expected_h,
        expected_w,
    )
    assert len(parsed.blocks) == expected_h * expected_w


def test_pewter_gym_exit_warps_are_last_map_sentinel(rom):
    """Gym exit doors resolve to `wLastMap` at runtime (ADR 0002) - the raw
    $FF byte must surface as `None`, never as a literal map ID 255."""
    pewter_gym = rom_maps.parse_map(rom, 54)
    assert len(pewter_gym.warps) == 2
    assert all(warp.dest_map is None for warp in pewter_gym.warps)


def test_brock_object_record_decodes_the_verified_encoding(rom):
    """`[sprite, y+4, x+4, movement, facing, flags]` plus trainer-flag
    bytes, per `pret/pokered data/maps/objects/PewterGym.asm`:
    `object_event 4, 1, SPRITE_SUPER_NERD, STAY, DOWN,
    TEXT_PEWTERGYM_BROCK, OPP_BROCK, 1`."""
    pewter_gym = rom_maps.parse_map(rom, 54)
    brock = next(obj for obj in pewter_gym.objects if obj.trainer is not None)
    assert (brock.x, brock.y) == (4, 1)
    assert brock.movement == 0xFF  # STAY
    assert brock.trainer == (234, 1)  # OPP_BROCK, trainer number 1


def test_pallet_town_connections_match_known_kanto_geography(rom):
    """Regression test for a bit-order bug: connection *records* compile in
    north/south/west/east authoring order (`macros/scripts/maps.asm`), not
    ascending bit-mask value - iterating in bit-value order silently
    swapped Pallet Town's real north/south neighbors."""
    pallet_town = rom_maps.parse_map(rom, 0)
    by_direction = {c.direction: c.dest_map for c in pallet_town.connections}
    assert by_direction == {"north": 12, "south": 32}  # Route 1, Route 21


def test_viridian_city_three_way_connections_match_known_geography(rom):
    viridian_city = rom_maps.parse_map(rom, 1)
    by_direction = {c.direction: c.dest_map for c in viridian_city.connections}
    assert by_direction == {
        "north": 13,
        "south": 12,
        "west": 33,
    }  # Route 2, Route 1, Route 22


def test_milestone_target_literals_match_the_roms_object_records(all_maps):
    """Regression guard for issue #98's hand-authored milestone -> object
    selection (`milestone_targets.py`): re-derives every milestone's
    `(x, y)` from this parser and checks it against the literals pinned in
    `milestones.py`'s `_MILESTONES`, so a `pokemon_red.gb` re-dump or an
    upstream layout change fails here instead of silently drifting."""
    resolved = resolve_target_coordinates(all_maps)
    pinned = {
        check.milestone.milestone_id: (
            check.milestone.target.target_x,
            check.milestone.target.target_y,
        )
        for check in milestones._MILESTONES
    }
    assert resolved == pinned
