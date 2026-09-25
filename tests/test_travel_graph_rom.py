"""ROM-backed sanity check that `travel_graph.build_milestone_travel_graph`
produces real, correct routes against this repo's own `pokemon_red.gb` -
gated behind the ROM file's presence, same pattern as `tests/
test_rom_maps.py`/`tests/test_navigation.py`. The search algorithm itself
is covered independently of the ROM by `tests/test_travel_graph.py`; this
file only checks that the real hop *data* built from #97's parser comes out
right for the milestones this ticket's incremental build covers."""

from pathlib import Path

import pytest

from jev_plays_pokemon import rom_maps
from jev_plays_pokemon import travel_graph as tg

ROM_PATH = Path(__file__).resolve().parent.parent / "pokemon_red.gb"

pytestmark = pytest.mark.skipif(
    not ROM_PATH.exists(), reason=f"{ROM_PATH} not present locally"
)

_MAP_PALLET_TOWN = 0
_MAP_PEWTER_CITY = 2
_MAP_CERULEAN_CITY = 3
_MAP_OAKS_LAB = 40
# The player's own house - the maps a fresh boot is standing on. See
# `test_the_graph_starts_where_a_fresh_boot_actually_leaves_the_player`.
_MAP_REDS_HOUSE_1F = 37
_MAP_REDS_HOUSE_2F = 38
_MAP_VIRIDIAN_MART = 42
_MAP_PEWTER_GYM = 54
_MAP_CERULEAN_GYM = 65
_MAP_BILLS_HOUSE = 88
_MAP_VERMILION_GYM = 92
_MAP_CELADON_GYM = 134
# Silph Co 2F: one of the 7 maps #96's "Out of Scope" section names as never
# getting a travel-graph route (its parsed ROM records disagreed with
# upstream pret/pokered during that ticket's research) - a permanently safe
# "no known route yet" example, unlike map 65 (Cerulean Gym), which #99
# brought into MILESTONE_MAP_IDS.
_MAP_NOT_YET_ROUTED = 207


@pytest.fixture(scope="module")
def maps() -> dict[int, rom_maps.RomMap]:
    rom = rom_maps.load_rom(ROM_PATH)
    return {map_id: rom_maps.parse_map(rom, map_id) for map_id in tg.MILESTONE_MAP_IDS}


@pytest.fixture(scope="module")
def graph(maps: dict[int, rom_maps.RomMap]) -> tg.TravelGraph:
    return tg.build_milestone_travel_graph(maps)


def test_pallet_town_to_oaks_lab_is_a_single_door_hop(graph):
    route = tg.find_route(graph, _MAP_PALLET_TOWN, _MAP_OAKS_LAB)
    assert route is not None
    assert len(route) == 1
    hop = route[0]
    assert hop.kind == "warp"
    assert (hop.from_x, hop.from_y) == (12, 11)
    assert (hop.to_x, hop.to_y) == (5, 11)


def test_oaks_lab_last_map_exit_resolves_back_to_pallet_towns_own_door(graph):
    """Regression for #101's core requirement: the exit's `LAST_MAP`
    destination must resolve to the exact tile the player walked in
    through, not a hardcoded map - independently verified live from RAM in
    `docs/research/gen1-map-coordinate-sources.md`."""
    route = tg.find_route(graph, _MAP_OAKS_LAB, _MAP_PALLET_TOWN)
    assert route is not None
    assert len(route) == 1
    assert (route[0].to_x, route[0].to_y) == (12, 11)


def test_pallet_town_to_viridian_mart_crosses_two_connection_legs(graph):
    """Pallet Town -> Route 1 -> Viridian City (2 `connection` hops), then
    the Poke Mart's own entrance door."""
    route = tg.find_route(graph, _MAP_PALLET_TOWN, _MAP_VIRIDIAN_MART)
    assert route is not None
    assert [hop.kind for hop in route] == ["connection", "connection", "warp"]
    assert route[-1].to_map == _MAP_VIRIDIAN_MART


def test_pallet_town_to_pewter_gym_crosses_multiple_outdoor_connections(graph):
    """ADR-0002's own named example: Pallet Town -> Pewter Gym crosses
    several outdoor route/city maps via `connection` hops before the final
    `warp_event` door into the gym."""
    route = tg.find_route(graph, _MAP_PALLET_TOWN, _MAP_PEWTER_GYM)
    assert route is not None
    assert len(route) == 5
    assert [hop.kind for hop in route] == [
        "connection",
        "connection",
        "connection",
        "connection",
        "warp",
    ]
    assert route[-1].to_map == _MAP_PEWTER_GYM
    assert route[-1].from_map == _MAP_PEWTER_CITY


def test_pallet_town_to_cerulean_gym_crosses_the_northbound_overland_corridor(graph):
    """#99: Pallet Town -> Route 1 -> Viridian City -> Route 2 -> Pewter
    City -> Route 3 -> Route 4 -> Cerulean City -> Cerulean Gym."""
    route = tg.find_route(graph, _MAP_PALLET_TOWN, _MAP_CERULEAN_GYM)
    assert route is not None
    assert route[-1].to_map == _MAP_CERULEAN_GYM
    assert route[-1].from_map == _MAP_CERULEAN_CITY


def test_cerulean_city_to_bills_house_crosses_routes_24_and_25(graph):
    route = tg.find_route(graph, _MAP_CERULEAN_CITY, _MAP_BILLS_HOUSE)
    assert route is not None
    assert route[-1].to_map == _MAP_BILLS_HOUSE


def test_cerulean_city_to_vermilion_gym_crosses_saffron_city(graph):
    route = tg.find_route(graph, _MAP_CERULEAN_CITY, _MAP_VERMILION_GYM)
    assert route is not None
    assert route[-1].to_map == _MAP_VERMILION_GYM


def test_cerulean_city_to_celadon_gym_crosses_saffron_city(graph):
    route = tg.find_route(graph, _MAP_CERULEAN_CITY, _MAP_CELADON_GYM)
    assert route is not None
    assert route[-1].to_map == _MAP_CELADON_GYM


def test_no_known_route_yet_for_a_milestone_outside_this_tickets_scope(graph):
    """Silph Co 2F isn't in `MILESTONE_MAP_IDS` (#96's own "Out of Scope"
    list) - the search must signal that cleanly, not raise."""
    assert tg.find_route(graph, _MAP_PALLET_TOWN, _MAP_NOT_YET_ROUTED) is None


def test_the_graph_starts_where_a_fresh_boot_actually_leaves_the_player(graph):
    """A fresh boot leaves the player controllable on Red's House 2F (`emulator.
    boot_past_intro`), not in Pallet Town, so the graph's first node has to be
    the bedroom's own map or the run's first milestone is unreachable from its
    first tile: `resolve_navigation_target` returns `None`, `decision.py` drops
    the navigation macro from the action space, and the turn has nothing left
    but raw buttons."""
    assert _MAP_REDS_HOUSE_2F in tg.MILESTONE_MAP_IDS
    assert _MAP_REDS_HOUSE_1F in tg.MILESTONE_MAP_IDS

    route = tg.find_route(graph, _MAP_REDS_HOUSE_2F, _MAP_OAKS_LAB)

    assert route is not None
    assert [(h.from_map, h.to_map) for h in route] == [
        (_MAP_REDS_HOUSE_2F, _MAP_REDS_HOUSE_1F),
        (_MAP_REDS_HOUSE_1F, _MAP_PALLET_TOWN),
        (_MAP_PALLET_TOWN, _MAP_OAKS_LAB),
    ]
    assert (route[0].from_x, route[0].from_y) == (7, 1)  # the stairs, from (3, 6)
    assert (route[1].from_x, route[1].from_y) == (2, 7)  # the front door
    assert (route[1].to_x, route[1].to_y) == (5, 5)  # the town tile it lands on


def test_reds_house_1f_resolves_its_last_map_door_to_the_towns_front_door(maps):
    """Red's House 1F has two entrances in scope - Pallet Town's front door and
    its own upstairs staircase - so "which map does this map's `LAST_MAP` door
    return to?" can't be answered by counting incoming warps: there are two
    candidates and the door gets dropped. Matching on the tile each candidate
    *lands on* separates them, and the door's exit must land back on the exact
    tile the player walked in through."""
    exit_hops = [
        hop
        for hop in tg.build_hops(maps, tg.MILESTONE_MAP_IDS)
        if hop.from_map == _MAP_REDS_HOUSE_1F and hop.to_map == _MAP_PALLET_TOWN
    ]

    assert [(h.from_x, h.from_y, h.to_x, h.to_y) for h in exit_hops] == [(2, 7, 5, 5)]
