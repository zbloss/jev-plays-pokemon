"""Plain unit tests for `travel_graph.py`'s search algorithm and hop-data
builders, over small hand-built fixture graphs/maps - no ROM, no PyBoy (see
that module's and ADR-0002's "Search"/"Scope" sections). These run always-
on: nothing here is gated behind `pokemon_red.gb`'s presence, unlike
`tests/test_travel_graph_rom.py`."""

import pytest

from jev_plays_pokemon import travel_graph as tg
from jev_plays_pokemon.rom_maps import MapConnection, MapWarp, RomMap

# Fictional map IDs for the pure-search fixtures below - deliberately not
# real `pret/pokered` map IDs, so these tests can't be confused for
# ROM-verified geography.
_MAP_A = 101
_MAP_B = 102
_MAP_C = 103
_MAP_D = 104
_MAP_UNREACHABLE = 199


def _hop(from_map: int, to_map: int, kind: str = "warp") -> tg.Hop:
    return tg.Hop(
        from_map=from_map,
        from_x=1,
        from_y=1,
        to_map=to_map,
        to_x=2,
        to_y=2,
        kind=kind,
    )


def _chain_graph() -> tg.TravelGraph:
    """A -> B -> C -> D, each hop reversible, plus an unreachable island."""
    graph = tg.TravelGraph()
    for src, dst in ((_MAP_A, _MAP_B), (_MAP_B, _MAP_C), (_MAP_C, _MAP_D)):
        graph.add_hop(_hop(src, dst, kind="connection"))
        graph.add_hop(_hop(dst, src, kind="connection"))
    return graph


def test_find_route_returns_empty_tuple_when_already_at_target():
    graph = _chain_graph()
    assert tg.find_route(graph, _MAP_A, _MAP_A) == ()


def test_find_route_finds_a_single_hop_route():
    graph = _chain_graph()
    route = tg.find_route(graph, _MAP_A, _MAP_B)
    assert route is not None
    assert len(route) == 1
    assert route[0].from_map == _MAP_A
    assert route[0].to_map == _MAP_B


def test_find_route_finds_the_shortest_multi_hop_route():
    graph = _chain_graph()
    route = tg.find_route(graph, _MAP_A, _MAP_D)
    assert route is not None
    assert [hop.from_map for hop in route] == [_MAP_A, _MAP_B, _MAP_C]
    assert [hop.to_map for hop in route] == [_MAP_B, _MAP_C, _MAP_D]


def test_find_route_prefers_the_shorter_of_two_paths():
    """A direct A->D shortcut alongside the A->B->C->D chain: BFS must pick
    the 1-hop shortcut, not the 3-hop chain."""
    graph = _chain_graph()
    graph.add_hop(_hop(_MAP_A, _MAP_D, kind="warp"))
    route = tg.find_route(graph, _MAP_A, _MAP_D)
    assert route is not None
    assert len(route) == 1
    assert route[0].kind == "warp"


def test_next_hop_picks_the_first_hop_of_a_multi_hop_route():
    graph = _chain_graph()
    hop = tg.next_hop(graph, _MAP_A, _MAP_D)
    assert hop is not None
    assert (hop.from_map, hop.to_map) == (_MAP_A, _MAP_B)


def test_next_hop_is_none_when_already_at_target():
    graph = _chain_graph()
    assert tg.next_hop(graph, _MAP_A, _MAP_A) is None


def test_find_route_signals_no_known_route_as_none_not_an_exception():
    graph = _chain_graph()
    assert tg.find_route(graph, _MAP_A, _MAP_UNREACHABLE) is None


def test_next_hop_signals_no_known_route_as_none_not_an_exception():
    graph = _chain_graph()
    assert tg.next_hop(graph, _MAP_A, _MAP_UNREACHABLE) is None


def test_find_route_from_an_unlisted_current_map_is_no_route_not_a_crash():
    """A map with no outgoing hops at all (never added to the graph) is a
    graceful no-route, not a `KeyError`."""
    graph = _chain_graph()
    assert tg.find_route(graph, 999999, _MAP_A) is None


# --- Hop-data builders: resolve_warp_destination / LAST_MAP / connections ---

# A small hand-built two-map fixture mirroring Oak's Lab <-> Pallet Town's
# real shape (an interior with a `LAST_MAP` exit, an outdoor map whose own
# warp table is what the exit resolves against) - values are invented, not
# read from the ROM, matching this file's "no ROM" scope.
_OUTDOOR_MAP_ID = 201
_INTERIOR_MAP_ID = 202

_OUTDOOR_MAP = RomMap(
    map_id=_OUTDOOR_MAP_ID,
    tileset_id=0,
    height_blocks=9,
    width_blocks=10,
    blocks=b"",
    objects=(),
    warps=(
        MapWarp(x=1, y=1, dest_warp=0, dest_map=999),  # unrelated warp
        MapWarp(x=9, y=9, dest_warp=0, dest_map=888),  # unrelated warp
        MapWarp(x=12, y=11, dest_warp=1, dest_map=_INTERIOR_MAP_ID),
    ),
    connections=(),
)

_INTERIOR_MAP = RomMap(
    map_id=_INTERIOR_MAP_ID,
    tileset_id=1,
    height_blocks=6,
    width_blocks=5,
    blocks=b"",
    objects=(),
    warps=(
        MapWarp(x=4, y=11, dest_warp=2, dest_map=None),  # LAST_MAP exit
        MapWarp(x=5, y=11, dest_warp=2, dest_map=None),  # LAST_MAP exit
    ),
    connections=(),
)

_ROM_MAPS = {_OUTDOOR_MAP_ID: _OUTDOOR_MAP, _INTERIOR_MAP_ID: _INTERIOR_MAP}


def test_resolve_warp_destination_for_an_ordinary_static_warp():
    warp = _OUTDOOR_MAP.warps[2]
    dest_map, dest_x, dest_y = tg.resolve_warp_destination(_ROM_MAPS, warp)
    assert (dest_map, dest_x, dest_y) == (_INTERIOR_MAP_ID, 5, 11)


def test_resolve_warp_destination_for_last_map_needs_entered_from():
    warp = _INTERIOR_MAP.warps[0]
    with pytest.raises(ValueError):
        tg.resolve_warp_destination(_ROM_MAPS, warp)


def test_resolve_warp_destination_resolves_last_map_via_entered_from():
    """The `LAST_MAP` exit's `dest_warp` index (2) looks up the *outdoor*
    map's own warp table at that index - which is the exact door the
    player walked in through, exactly matching the real Oak's Lab <->
    Pallet Town pairing this fixture mirrors."""
    warp = _INTERIOR_MAP.warps[0]
    dest_map, dest_x, dest_y = tg.resolve_warp_destination(
        _ROM_MAPS, warp, entered_from=_OUTDOOR_MAP_ID
    )
    assert (dest_map, dest_x, dest_y) == (_OUTDOOR_MAP_ID, 12, 11)


def test_build_hops_resolves_last_map_edges_without_a_hardcoded_destination():
    """The interior's own `LAST_MAP` exits carry no destination map at all
    in their raw data - `build_hops` must discover the outdoor map
    (`_OUTDOOR_MAP_ID`) itself, from the scoped ROM data, not from any
    value this test or `travel_graph.py` hand-declares.

    Only the door tile the player can actually be *landed on* gets an exit
    hop (see `_find_entrance_map`), which is why this is one hop and not the
    two `LAST_MAP` records the interior declares: the interior's second
    `LAST_MAP` tile is the other half of the same door, and nothing in scope
    lands on it, so a hop out of it would be a claim about a `wLastMap` value
    no walk can ever produce."""
    map_ids = frozenset({_OUTDOOR_MAP_ID, _INTERIOR_MAP_ID})
    hops = tg.build_hops(_ROM_MAPS, map_ids)

    forward = [
        h
        for h in hops
        if h.from_map == _OUTDOOR_MAP_ID and h.to_map == _INTERIOR_MAP_ID
    ]
    assert len(forward) == 1
    assert (forward[0].from_x, forward[0].from_y) == (12, 11)
    assert (forward[0].to_x, forward[0].to_y) == (5, 11)

    backward = [
        h
        for h in hops
        if h.from_map == _INTERIOR_MAP_ID and h.to_map == _OUTDOOR_MAP_ID
    ]
    assert len(backward) == 1  # the landing half of the door, and only that
    assert (backward[0].from_x, backward[0].from_y) == (5, 11)
    assert (backward[0].to_x, backward[0].to_y) == (12, 11)


# An interior with *two* entrances, mirroring Red's House 1F: one door from
# the town and one staircase from its own upstairs floor, both landing on this
# map, both with a `LAST_MAP` record on the tile they land on. Counting maps
# that warp into the interior finds two candidates and learns nothing; the
# tile a walk can be dropped onto identifies the door it came in through.
_TWO_ENTRANCE_INTERIOR_ID = 203
_UPSTAIRS_MAP_ID = 204
_TWO_ENTRANCE_TOWN_ID = 205
_SECOND_TOWN_ID = 206

_TWO_ENTRANCE_TOWN = RomMap(
    map_id=_TWO_ENTRANCE_TOWN_ID,
    tileset_id=0,
    height_blocks=9,
    width_blocks=10,
    blocks=b"",
    objects=(),
    warps=(MapWarp(x=5, y=5, dest_warp=0, dest_map=_TWO_ENTRANCE_INTERIOR_ID),),
    connections=(),
)

_TWO_ENTRANCE_INTERIOR = RomMap(
    map_id=_TWO_ENTRANCE_INTERIOR_ID,
    tileset_id=2,
    height_blocks=4,
    width_blocks=4,
    blocks=b"",
    objects=(),
    warps=(
        MapWarp(x=2, y=7, dest_warp=0, dest_map=None),  # the town's front door
        MapWarp(x=7, y=1, dest_warp=0, dest_map=_UPSTAIRS_MAP_ID),  # up the stairs
    ),
    connections=(),
)

_UPSTAIRS_MAP = RomMap(
    map_id=_UPSTAIRS_MAP_ID,
    tileset_id=2,
    height_blocks=4,
    width_blocks=4,
    blocks=b"",
    objects=(),
    warps=(MapWarp(x=7, y=1, dest_warp=1, dest_map=_TWO_ENTRANCE_INTERIOR_ID),),
    connections=(),
)

_TWO_ENTRANCE_ROM_MAPS = {
    _TWO_ENTRANCE_TOWN_ID: _TWO_ENTRANCE_TOWN,
    _TWO_ENTRANCE_INTERIOR_ID: _TWO_ENTRANCE_INTERIOR,
    _UPSTAIRS_MAP_ID: _UPSTAIRS_MAP,
}
_TWO_ENTRANCE_MAP_IDS = frozenset(_TWO_ENTRANCE_ROM_MAPS)


def test_find_entrance_map_picks_the_map_whose_warp_lands_on_this_exact_tile():
    """Two maps warp into this interior, so "which map is the `LAST_MAP`
    door's `wLastMap`?" has no answer in a count of incoming warps - it has
    one in *where* each of them drops the player. The front door's tile is
    the town's landing tile, the staircase's is the upstairs map's, and each
    `LAST_MAP` record resolves to its own counterpart."""
    front_door, staircase = _TWO_ENTRANCE_INTERIOR.warps

    assert (
        tg._find_entrance_map(
            _TWO_ENTRANCE_ROM_MAPS,
            _TWO_ENTRANCE_MAP_IDS,
            _TWO_ENTRANCE_INTERIOR_ID,
            front_door,
        )
        == _TWO_ENTRANCE_TOWN_ID
    )
    assert (
        tg._find_entrance_map(
            _TWO_ENTRANCE_ROM_MAPS,
            _TWO_ENTRANCE_MAP_IDS,
            _TWO_ENTRANCE_INTERIOR_ID,
            staircase,
        )
        == _UPSTAIRS_MAP_ID
    )


def test_find_entrance_map_refuses_to_guess_when_two_maps_land_in_the_same_spot():
    """Two in-scope maps landing on one tile is a real ambiguity about
    `wLastMap`, and a guessed map would produce a hop to a tile the player
    can never arrive at - `None` (and therefore no hop) is the honest answer."""
    second_town = RomMap(
        map_id=_SECOND_TOWN_ID,
        tileset_id=0,
        height_blocks=9,
        width_blocks=10,
        blocks=b"",
        objects=(),
        warps=(MapWarp(x=1, y=1, dest_warp=0, dest_map=_TWO_ENTRANCE_INTERIOR_ID),),
        connections=(),
    )
    rom_maps = {**_TWO_ENTRANCE_ROM_MAPS, _SECOND_TOWN_ID: second_town}
    front_door = _TWO_ENTRANCE_INTERIOR.warps[0]

    assert (
        tg._find_entrance_map(
            rom_maps,
            _TWO_ENTRANCE_MAP_IDS | {_SECOND_TOWN_ID},
            _TWO_ENTRANCE_INTERIOR_ID,
            front_door,
        )
        is None
    )


def test_build_hops_gives_a_two_entrance_interior_one_door_back_out():
    """The consequence the two tests above exist for: the front door gets its
    exit hop, so a route out of a two-entrance map exists even though a
    count-based lookup would have dropped the door for being ambiguous."""
    hops = tg.build_hops(_TWO_ENTRANCE_ROM_MAPS, _TWO_ENTRANCE_MAP_IDS)
    leaving = {
        (h.from_x, h.from_y, h.to_map, h.to_x, h.to_y)
        for h in hops
        if h.from_map == _TWO_ENTRANCE_INTERIOR_ID
    }

    assert (2, 7, _TWO_ENTRANCE_TOWN_ID, 5, 5) in leaving
    assert (7, 1, _UPSTAIRS_MAP_ID, 7, 1) in leaving


def test_build_hops_skips_warps_whose_destination_is_out_of_scope():
    map_ids = frozenset({_OUTDOOR_MAP_ID, _INTERIOR_MAP_ID})
    hops = tg.build_hops(_ROM_MAPS, map_ids)
    assert all(h.to_map != 999 for h in hops)


def test_build_hops_skips_same_map_warps():
    """A same-map teleport-style warp (Saffron Gym's puzzle tiles, per
    ADR-0002) never produces a hop - it isn't a cross-map transition."""
    puzzle_map_id = 301
    puzzle_map = RomMap(
        map_id=puzzle_map_id,
        tileset_id=0,
        height_blocks=1,
        width_blocks=1,
        blocks=b"",
        objects=(),
        warps=(MapWarp(x=1, y=1, dest_warp=0, dest_map=puzzle_map_id),),
        connections=(),
    )
    hops = tg.build_hops({puzzle_map_id: puzzle_map}, frozenset({puzzle_map_id}))
    assert hops == ()


def test_build_travel_graph_is_searchable_end_to_end():
    map_ids = frozenset({_OUTDOOR_MAP_ID, _INTERIOR_MAP_ID})
    graph = tg.build_travel_graph(_ROM_MAPS, map_ids)
    route = tg.find_route(graph, _OUTDOOR_MAP_ID, _INTERIOR_MAP_ID)
    assert route is not None
    assert len(route) == 1
    assert (route[0].to_x, route[0].to_y) == (5, 11)


# --- Connection hops ---

_NORTH_MAP_ID = 401
_SOUTH_MAP_ID = 402

_SOUTH_MAP = RomMap(
    map_id=_SOUTH_MAP_ID,
    tileset_id=0,
    height_blocks=9,
    width_blocks=10,  # 20 tiles wide
    blocks=b"",
    objects=(),
    warps=(),
    connections=(
        MapConnection(
            direction="north", dest_map=_NORTH_MAP_ID, y_alignment=35, x_alignment=0
        ),
    ),
)

_NORTH_MAP = RomMap(
    map_id=_NORTH_MAP_ID,
    tileset_id=0,
    height_blocks=18,
    width_blocks=10,  # 20 tiles wide, 36 tiles tall
    blocks=b"",
    objects=(),
    warps=(),
    connections=(
        MapConnection(
            direction="south", dest_map=_SOUTH_MAP_ID, y_alignment=0, x_alignment=0
        ),
    ),
)


def test_connection_hop_snaps_the_perpendicular_axis_to_the_shared_edge():
    """North connection: leaving from the south map's row 0 (its north
    edge), landing on the north map's south edge (`y_alignment` = 35, i.e.
    `height_tiles - 1`) - matching this repo's own Pallet Town <-> Route 1
    connection (`tests/test_rom_maps.py`'s alignment-byte test)."""
    hop = tg.connection_hop(
        _SOUTH_MAP_ID, _SOUTH_MAP, _SOUTH_MAP.connections[0], _NORTH_MAP
    )
    assert hop.from_y == 0
    assert hop.to_y == 35
    assert hop.to_map == _NORTH_MAP_ID


def test_connection_hop_carries_the_along_edge_axis_with_no_offset_when_aligned():
    """Both maps are the same width with `x_alignment = 0`: the along-edge
    (X) coordinate must be identical on both sides."""
    hop = tg.connection_hop(
        _SOUTH_MAP_ID, _SOUTH_MAP, _SOUTH_MAP.connections[0], _NORTH_MAP
    )
    assert hop.from_x == hop.to_x


def test_connection_hop_applies_a_nonzero_along_edge_offset():
    """A wider destination map, offset by 10 tiles (mirrors this repo's
    real Viridian City <-> Route 1 pairing): `to_x = from_x + x_alignment`."""
    wide_map_id = 403
    wide_map = RomMap(
        map_id=wide_map_id,
        tileset_id=0,
        height_blocks=18,
        width_blocks=20,  # 40 tiles wide
        blocks=b"",
        objects=(),
        warps=(),
        connections=(),
    )
    connection = MapConnection(
        direction="north", dest_map=wide_map_id, y_alignment=35, x_alignment=10
    )
    hop = tg.connection_hop(_SOUTH_MAP_ID, _SOUTH_MAP, connection, wide_map)
    assert hop.to_x == hop.from_x + 10


def test_build_hops_includes_bidirectional_connection_hops():
    map_ids = frozenset({_NORTH_MAP_ID, _SOUTH_MAP_ID})
    rom_maps = {_NORTH_MAP_ID: _NORTH_MAP, _SOUTH_MAP_ID: _SOUTH_MAP}
    hops = tg.build_hops(rom_maps, map_ids)
    assert {(h.from_map, h.to_map) for h in hops} == {
        (_SOUTH_MAP_ID, _NORTH_MAP_ID),
        (_NORTH_MAP_ID, _SOUTH_MAP_ID),
    }
