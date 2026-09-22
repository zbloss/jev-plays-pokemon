"""The travel graph: hop data plus cross-map search, per ADR-0002.

Per `CONTEXT.md`'s "Travel graph"/"Hop" entries: a hand-authored, reusable
graph of hops - warp doors (`warp_event`) and map-edge connections
(`connection`) - that the navigation macro's cross-map routing (#102)
searches, statelessly, on every invocation to find the next hop toward the
current milestone's target map. A hop is "leave the current map at tile X,
arrive at map M near tile Y" - the unit both warp doors and map-edge
connections are modeled as, uniformly.

## Hop tile data

`build_hops`/`build_travel_graph` derive hop tiles from `rom_maps.py`'s
(#97) parsed warp/connection records - no hand-typed coordinates:

- **Warps.** A warp's landing tile is always "the Nth entry in the
  destination map's own warp table" (`N` = the warp's own `dest_warp`
  field) - this is how `pret/pokered`'s own `LoadDestinationWarpPosition`
  resolves it at runtime (`docs/research/pokered-warp-data.md`), and it
  holds regardless of whether the destination map is known statically or
  is the `LAST_MAP` sentinel. `resolve_warp_destination` implements this
  lookup; `LAST_MAP` warps need an `entered_from` map supplied (see below).
- **`LAST_MAP` resolution.** A `LAST_MAP` exit door's destination map isn't
  in the warp record - it's `wLastMap`, "whichever map the player warped in
  from" - but within this incrementally-built, scoped graph, that map is
  always discoverable: it's whichever *other* in-scope map has a warp
  targeting the exit door's own map. `_find_entrance_map` does exactly that
  reverse lookup, so `LAST_MAP` edges resolve to a real, ROM-derived tile
  (never a hand-typed destination) - verified end to end against this
  repo's own ROM: Oak's Lab's `LAST_MAP` exit (`dest_warp=2`) resolves to
  Pallet Town's warp index 2, `(12, 11)` - the exact door the player walked
  in through, and the same tile `docs/research/gen1-map-coordinate-
  sources.md` independently verified live from RAM.
- **Connections.** `pret/pokered`'s `home/overworld.asm`
  (`CheckMapConnections`) computes a crossed map edge's landing tile from
  two signed alignment bytes now decoded onto `rom_maps.MapConnection`
  (see that class's docstring): the axis perpendicular to the edge snaps to
  a fixed tile row/column on the destination map, and the along-edge axis
  carries the source coordinate over plus a signed offset. `connection_hop`
  picks the midpoint of both maps' overlapping valid range along that axis
  - any tile in that range is an equally valid, real crossing point (this
  isn't a single fixed door tile the way a warp is), verified against this
  repo's ROM for the Pallet Town <-> Route 1 <-> Viridian City <-> Route 2
  <-> Pewter City chain (see `tests/test_travel_graph_rom.py`).

## Scope

`build_milestone_travel_graph` builds hops for a small, explicitly-listed
set of map IDs - not the whole ~223-map graph - per ADR-0002's incremental-
build decision ("this ticket doesn't need to build routes for every
milestone"). It currently covers Pallet Town -> Oak's Lab (`got_starter`/
`got_pokedex`), Pallet Town -> Viridian City -> Viridian Mart
(`got_oaks_parcel`), the full Pallet Town -> Route 1 -> Viridian City ->
Route 2 -> Pewter City -> Pewter Gym chain (`boulder_badge`) - the
multi-hop, connection-crossing case ADR-0002 names by example - and (#99)
the overland corridor onward from Pewter City through Route 3/Route 4 to
Cerulean City and Cerulean Gym (`cascade_badge`), Cerulean -> Route 24 ->
Route 25 -> Bill's House (`got_ss_ticket`), and Cerulean -> Route 5 ->
Saffron City -> Route 6 -> Vermilion City/Gym (`thunder_badge`) and
Saffron -> Route 7 -> Celadon City/Gym (`rainbow_badge`) - all real,
walkable land routes (no Surf/water crossing), confirmed against this
repo's own ROM data. (#114) adds Viridian Gym itself (`earth_badge`) - a
single warp hop off Viridian City, already in scope alongside it, which
was all that was missing for `build_hops` to produce that edge at all.
Milestones outside this set (e.g. `soul_badge`)
aren't yet reachable in the graph; `next_hop`/`find_route` signal that as
`None`, per ADR-0002's statelessness (#102 treats a `None` route as a
graceful no-op, matching today's same-map-only behavior), rather than
raising - a gap here is expected,
incremental-build territory, not a bug.

## A known false edge: Route 4 is two maps wearing one name

Every hop here is derived from warp/connection *records*, which say where
a map edge or door lands - not whether its two endpoints are reachable
from each other on foot. Route 4 is the case where that distinction
bites: `tileset_collision.py` decodes its whole 90x18 tile grid and
route `x=20`-`23` is an unbroken wall from `y=0` to `y=17`, splitting the
map into a west pocket (`x=4`-`19`, where Route 3's south connection
lands) and an east side (`x=24`+, holding the `(89, 8)` Cerulean City
connection). Confirmed live, not just decoded: PyBoy's own
`game_area_collision()` agrees tile-for-tile, and the on-screen A* walks
the pocket to exactly its east edge and stops.

So the graph's `Route 3 -> Route 4 -> Cerulean City` route is only
half-true - it names real hops that no ordinary walk can join end to end.
Route 4's two warps into Mt Moon (`(18, 5)` on the pocket side, `(24, 5)`
on the east side) are the actual connector, and Mt Moon is a three-floor
dungeon whose floors the last-mile A* can plan inside but not
cross-solve, so crossing it is a scripted concern, not graph routing.
Fixing the graph to say so is a separate decision from this ticket's
scope (see the fixture docs in `tests/test_navigation.py` for the walked
route); what matters here is that a route this module returns is a
sequence of real hops, not a promise that each hop's landing tile is
walkable from wherever the previous hop left the player.

## Search

`find_route`/`next_hop` run a plain BFS (unweighted - every hop costs 1,
per ADR-0002: "BFS/Dijkstra - cheap at this graph's size") from a current
map to a target map, returning the shortest hop sequence (or just its first
hop). Nothing is cached or persisted: each call re-searches from scratch,
matching `execute_navigation_macro`'s existing stateless re-plan-every-step
design (ADR-0002's "Statelessness" section) - if the player gets knocked
off course, the very next call just searches again from wherever they
actually ended up.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from jev_plays_pokemon.rom_maps import MapConnection, MapWarp, RomMap


@dataclass(frozen=True)
class Hop:
    """One travel-graph edge: leave `from_map` at `(from_x, from_y)`,
    arrive at `to_map` near `(to_x, to_y)`. `kind` is "warp" or
    "connection" - informational only, the search treats both alike."""

    from_map: int
    from_x: int
    from_y: int
    to_map: int
    to_x: int
    to_y: int
    kind: str


class TravelGraph:
    """An adjacency-list graph of `Hop`s, keyed by the map each hop leaves
    from. Directed: a bidirectional door/connection is two separate hops
    (one per direction), added separately."""

    def __init__(self) -> None:
        self._hops_by_map: dict[int, list[Hop]] = {}

    def add_hop(self, hop: Hop) -> None:
        self._hops_by_map.setdefault(hop.from_map, []).append(hop)

    def hops_from(self, map_id: int) -> tuple[Hop, ...]:
        return tuple(self._hops_by_map.get(map_id, ()))


def find_route(
    graph: TravelGraph, current_map: int, target_map: int
) -> tuple[Hop, ...] | None:
    """The shortest (fewest-hops) sequence of hops from `current_map` to
    `target_map`, or `None` if the graph doesn't (yet) connect them.

    `()` (an empty route) iff `current_map == target_map` - already there,
    no hop needed.
    """
    if current_map == target_map:
        return ()

    visited = {current_map}
    predecessor: dict[int, Hop] = {}
    queue: deque[int] = deque([current_map])

    while queue:
        here = queue.popleft()
        for hop in graph.hops_from(here):
            if hop.to_map in visited:
                continue
            visited.add(hop.to_map)
            predecessor[hop.to_map] = hop
            if hop.to_map == target_map:
                route = [hop]
                node = hop.from_map
                while node != current_map:
                    prior = predecessor[node]
                    route.append(prior)
                    node = prior.from_map
                route.reverse()
                return tuple(route)
            queue.append(hop.to_map)

    return None


def next_hop(graph: TravelGraph, current_map: int, target_map: int) -> Hop | None:
    """The first hop on the shortest route from `current_map` to
    `target_map`, or `None` if already there or if no route is known yet -
    the "no known route" signal #102's macro treats as a graceful no-op."""
    route = find_route(graph, current_map, target_map)
    if not route:
        return None
    return route[0]


def resolve_warp_destination(
    rom_maps: dict[int, RomMap], warp: MapWarp, *, entered_from: int | None = None
) -> tuple[int, int, int]:
    """`(dest_map, dest_x, dest_y)` for `warp`, per `pret/pokered`'s own
    resolution rule: the landing tile is always the `warp.dest_warp`-th
    entry in the destination map's own warp table (see module docstring).

    For an ordinary warp (`warp.dest_map` set), that destination map is
    used directly. For a `LAST_MAP` warp (`warp.dest_map is None`),
    `entered_from` - the map the player is presumed to have warped in
    from - stands in for it; omitting `entered_from` on a `LAST_MAP` warp
    is a caller error (there is no destination to resolve), not a "no
    route" case, so this raises rather than returning `None`.
    """
    dest_map = warp.dest_map if warp.dest_map is not None else entered_from
    if dest_map is None:
        raise ValueError(
            "warp destination is the LAST_MAP sentinel; an `entered_from` "
            "map is required to resolve it"
        )
    dest_table = rom_maps[dest_map].warps
    dest = dest_table[warp.dest_warp]
    return dest_map, dest.x, dest.y


def _find_entrance_map(
    rom_maps: dict[int, RomMap], map_ids: frozenset[int], target_map: int
) -> int | None:
    """Which map, among `map_ids`, has a warp whose (statically-known)
    destination is `target_map` - i.e. which map a `LAST_MAP` exit door on
    `target_map` returns to, discovered from the scoped ROM data itself
    rather than hand-declared. `None` if no in-scope map's warps target it
    (the scoped graph doesn't cover this map's entrance yet)."""
    candidates = {
        map_id
        for map_id in map_ids
        for warp in rom_maps[map_id].warps
        if warp.dest_map == target_map
    }
    if len(candidates) != 1:
        return None
    return next(iter(candidates))


def _tile_span(height_blocks: int, width_blocks: int) -> tuple[int, int]:
    """A map's (height, width) in tiles - a block is 2x2 tiles."""
    return height_blocks * 2, width_blocks * 2


def connection_hop(
    from_map_id: int, from_map: RomMap, connection: MapConnection, to_map: RomMap
) -> Hop:
    """A `Hop` for one map-edge `connection`, per the alignment arithmetic
    documented on `rom_maps.MapConnection`.

    The perpendicular axis snaps to a fixed edge row/column; the along-edge
    axis is picked at the midpoint of both maps' overlapping valid range
    (any tile in that range is an equally real crossing point - a
    connection isn't a single fixed door tile the way a warp is).
    """
    from_height, from_width = _tile_span(from_map.height_blocks, from_map.width_blocks)
    to_height, to_width = _tile_span(to_map.height_blocks, to_map.width_blocks)

    if connection.direction in ("north", "south"):
        offset = connection.x_alignment
        lo = max(0, -offset)
        hi = min(from_width - 1, to_width - 1 - offset)
        from_along = (lo + hi) // 2
        to_along = from_along + offset
        from_perp = 0 if connection.direction == "north" else from_height - 1
        to_perp = connection.y_alignment
        from_x, from_y = from_along, from_perp
        to_x, to_y = to_along, to_perp
    else:  # "east" | "west"
        offset = connection.y_alignment
        lo = max(0, -offset)
        hi = min(from_height - 1, to_height - 1 - offset)
        from_along = (lo + hi) // 2
        to_along = from_along + offset
        from_perp = from_width - 1 if connection.direction == "east" else 0
        to_perp = connection.x_alignment
        from_x, from_y = from_perp, from_along
        to_x, to_y = to_perp, to_along

    return Hop(
        from_map=from_map_id,
        from_x=from_x,
        from_y=from_y,
        to_map=connection.dest_map,
        to_x=to_x,
        to_y=to_y,
        kind="connection",
    )


def build_hops(rom_maps: dict[int, RomMap], map_ids: frozenset[int]) -> tuple[Hop, ...]:
    """Every hop leaving a map in `map_ids`, derived from `rom_maps`'
    warp/connection records - both endpoints must be in `map_ids` (scoping
    the graph to only what's needed, per ADR-0002). Warps whose destination
    is their own map (Saffron Gym's teleport-tile puzzle) are skipped -
    out of scope per ADR-0002, and not a cross-map hop regardless."""
    hops: list[Hop] = []
    for map_id in map_ids:
        rom_map = rom_maps[map_id]

        for warp in rom_map.warps:
            if warp.dest_map == map_id:
                continue
            if warp.dest_map is not None:
                if warp.dest_map not in map_ids:
                    continue  # destination out of scope - not a needed hop.
                entered_from = None
            else:
                entered_from = _find_entrance_map(rom_maps, map_ids, map_id)
                if entered_from is None:
                    continue  # LAST_MAP with no in-scope entrance yet.
            dest_map, dest_x, dest_y = resolve_warp_destination(
                rom_maps, warp, entered_from=entered_from
            )
            hops.append(
                Hop(
                    from_map=map_id,
                    from_x=warp.x,
                    from_y=warp.y,
                    to_map=dest_map,
                    to_x=dest_x,
                    to_y=dest_y,
                    kind="warp",
                )
            )

        for connection in rom_map.connections:
            if connection.dest_map not in map_ids:
                continue
            hops.append(
                connection_hop(
                    map_id, rom_map, connection, rom_maps[connection.dest_map]
                )
            )

    return tuple(hops)


def build_travel_graph(
    rom_maps: dict[int, RomMap], map_ids: frozenset[int]
) -> TravelGraph:
    """A `TravelGraph` over exactly `build_hops(rom_maps, map_ids)`."""
    graph = TravelGraph()
    for hop in build_hops(rom_maps, map_ids):
        graph.add_hop(hop)
    return graph


# Map IDs, per `constants/map_constants.asm`'s `const_def` position (same
# source `milestones.py` uses). The outdoor maps here are exactly the ones
# needed to route Pallet Town -> Pewter City (ADR-0002's own named example)
# plus Pallet Town -> Viridian City for the Viridian Mart leg, and (#99)
# onward to Cerulean/Bill's House/Vermilion/Celadon - see module docstring's
# "Scope" section.
_MAP_PALLET_TOWN = 0
_MAP_VIRIDIAN_CITY = 1
_MAP_PEWTER_CITY = 2
_MAP_CERULEAN_CITY = 3
_MAP_VERMILION_CITY = 5
_MAP_CELADON_CITY = 6
_MAP_SAFFRON_CITY = 10
_MAP_ROUTE_1 = 12
_MAP_ROUTE_2 = 13
_MAP_ROUTE_3 = 14
_MAP_ROUTE_4 = 15
_MAP_ROUTE_5 = 16
_MAP_ROUTE_6 = 17
_MAP_ROUTE_7 = 18
_MAP_ROUTE_24 = 35
_MAP_ROUTE_25 = 36
_MAP_OAKS_LAB = 40
_MAP_VIRIDIAN_MART = 42
_MAP_VIRIDIAN_GYM = 45
_MAP_PEWTER_GYM = 54
_MAP_CERULEAN_GYM = 65
_MAP_BILLS_HOUSE = 88
_MAP_VERMILION_GYM = 92
_MAP_CELADON_GYM = 134

MILESTONE_MAP_IDS: frozenset[int] = frozenset(
    {
        _MAP_PALLET_TOWN,
        _MAP_VIRIDIAN_CITY,
        _MAP_PEWTER_CITY,
        _MAP_CERULEAN_CITY,
        _MAP_VERMILION_CITY,
        _MAP_CELADON_CITY,
        _MAP_SAFFRON_CITY,
        _MAP_ROUTE_1,
        _MAP_ROUTE_2,
        _MAP_ROUTE_3,
        _MAP_ROUTE_4,
        _MAP_ROUTE_5,
        _MAP_ROUTE_6,
        _MAP_ROUTE_7,
        _MAP_ROUTE_24,
        _MAP_ROUTE_25,
        _MAP_OAKS_LAB,
        _MAP_VIRIDIAN_MART,
        _MAP_VIRIDIAN_GYM,
        _MAP_PEWTER_GYM,
        _MAP_CERULEAN_GYM,
        _MAP_BILLS_HOUSE,
        _MAP_VERMILION_GYM,
        _MAP_CELADON_GYM,
    }
)


def build_milestone_travel_graph(rom_maps: dict[int, RomMap]) -> TravelGraph:
    """The travel graph scoped to `MILESTONE_MAP_IDS` - the incrementally-
    built subset of the 14 milestone maps' routes this ticket covers (see
    module docstring's "Scope" section). `rom_maps` is `rom_maps.
    parse_all_maps`'s output (or any dict covering at least
    `MILESTONE_MAP_IDS`)."""
    return build_travel_graph(rom_maps, MILESTONE_MAP_IDS)
