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
_MAP_OAKS_LAB = 40
_MAP_VIRIDIAN_MART = 42
_MAP_PEWTER_GYM = 54
_MAP_CASCADE_GYM_NOT_YET_ROUTED = 65


@pytest.fixture(scope="module")
def graph() -> tg.TravelGraph:
    rom = rom_maps.load_rom(ROM_PATH)
    all_maps = {
        map_id: rom_maps.parse_map(rom, map_id) for map_id in tg.MILESTONE_MAP_IDS
    }
    return tg.build_milestone_travel_graph(all_maps)


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


def test_no_known_route_yet_for_a_milestone_outside_this_tickets_scope(graph):
    """Cascade Badge's gym isn't in `MILESTONE_MAP_IDS` yet (ADR-0002's
    incremental build) - the search must signal that cleanly, not raise."""
    assert (
        tg.find_route(graph, _MAP_PALLET_TOWN, _MAP_CASCADE_GYM_NOT_YET_ROUTED) is None
    )
