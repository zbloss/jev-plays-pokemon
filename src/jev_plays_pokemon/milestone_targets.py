"""The milestone-id -> ROM object-record selection (issue #98).

`rom_maps.py` decodes every map's object records; it has no opinion on which
record is *the objective* for a given milestone, since several maps carry
more than one and the choice is a judgment call (`docs/research/gen1-map-
coordinate-sources.md` section 2's "OAK substring trap" is the cautionary
example: naive substring matching resolved `got_starter` to the rival
instead of a starter ball). This module is that 14-line judgment call, one
entry per `milestones.py` milestone, made explicit and reviewable instead of
inferred.

Selection is by object-record *index* - each object's declared position in
`pret/pokered`'s `data/maps/objects/<Map>.asm` (`object_event` lines, in
source order), which `rom_maps.parse_map` preserves byte-for-byte since it
walks the compiled records in the same order they were assembled in. Indices
below were read directly off this repo's `pokemon_red.gb` via
`rom_maps.parse_map` and cross-checked against the upstream `.asm` source
(see each entry's comment); `tests/test_rom_maps.py` re-derives them from
the ROM and asserts they still match the coordinates hardcoded into
`milestones.py`, so a `pokemon_red.gb` re-dump or an upstream layout change
would fail loudly there rather than silently drift.

Two entries are script-stateful/dynamic rather than a single fixed NPC/item,
per the issue: `got_starter` and `got_pokedex` (see their comments below for
the one-line rationale each).
"""

from __future__ import annotations

from dataclasses import dataclass

from jev_plays_pokemon.rom_maps import RomMap

_MAP_OAKS_LAB = 40
_MAP_VIRIDIAN_MART = 42
_MAP_PEWTER_GYM = 54
_MAP_CERULEAN_GYM = 65
_MAP_BILLS_HOUSE = 88
_MAP_VERMILION_GYM = 92
_MAP_CELADON_GYM = 134
_MAP_POKEMON_TOWER_7F = 148
_MAP_FUCHSIA_GYM = 157
_MAP_SAFFRON_GYM = 178
_MAP_CINNABAR_GYM = 166
_MAP_VIRIDIAN_GYM = 45
_MAP_CHAMPIONS_ROOM = 120


@dataclass(frozen=True)
class _ObjectSelection:
    map_id: int
    object_index: int
    rationale: str


# milestone_id -> which of that milestone's map's `RomMap.objects` entries is
# the objective. Object indices are 0-based, matching declaration order in
# `pret/pokered`'s `data/maps/objects/<Map>.asm` (verified per-entry below by
# fetching that file); `resolve_target_coordinates` reads `.x`/`.y` off the
# selected record rather than any literal coordinate.
MILESTONE_OBJECT_SELECTIONS: dict[str, _ObjectSelection] = {
    "got_starter": _ObjectSelection(
        map_id=_MAP_OAKS_LAB,
        object_index=3,  # OaksLab.asm object_event #4: OAKSLAB_BULBASAUR_POKE_BALL
        rationale=(
            "The player's actual pick isn't tracked by any GameState field "
            "this repo reads, so one of the three starter balls has to be a "
            "static choice; Bulbasaur is picked because Grass resists "
            "Misty's Water and hits Brock's Rock/Ground super-effectively, "
            "the conventional 'easy start' choice for the first two gyms."
        ),
    ),
    "got_oaks_parcel": _ObjectSelection(
        map_id=_MAP_VIRIDIAN_MART,
        object_index=2,  # ViridianMart.asm object_event #3: VIRIDIANMART_COOLTRAINER_M
        rationale="The only NPC in the mart that hands over Oak's Parcel.",
    ),
    "got_pokedex": _ObjectSelection(
        map_id=_MAP_OAKS_LAB,
        object_index=4,  # OaksLab.asm object_event #5: OAKSLAB_OAK1
        rationale=(
            "pret/pokered scripts/OaksLab.asm: delivering the parcel talks "
            "to the OAK1 object (`OaksLabOak1Text`'s `.got_parcel` branch), "
            "which removes the parcel and sets EVENT_GOT_POKEDEX; OAK2 is "
            "Oak's later story-beat position and OAKSLAB_RIVAL (this map's "
            "other script-adjacent object) is a consequence of that same "
            "scene, not the trigger - so OAK1 is correct even though "
            "`docs/research/gen1-map-coordinate-sources.md`'s table guesses "
            "the rival."
        ),
    ),
    "boulder_badge": _ObjectSelection(
        map_id=_MAP_PEWTER_GYM,
        object_index=0,  # PewterGym.asm object_event #1: PEWTERGYM_BROCK
        rationale="Gym leaders are conventionally the first object_event in pret/pokered's map files; Brock is this map's only OPP_BROCK trainer record.",
    ),
    "cascade_badge": _ObjectSelection(
        map_id=_MAP_CERULEAN_GYM,
        object_index=0,  # CeruleanGym.asm object_event #1: CERULEANGYM_MISTY
        rationale="This map's only OPP_MISTY trainer record.",
    ),
    "got_ss_ticket": _ObjectSelection(
        map_id=_MAP_BILLS_HOUSE,
        object_index=1,  # BillsHouse.asm object_event #2: BILLSHOUSE_BILL_SS_TICKET
        rationale=(
            "Bill's house has three records at two tiles (his cat/monster "
            "form and post-ticket dialogue both sit at (6,5)); index 1 is "
            "the human-Bill-hands-over-the-ticket text specifically."
        ),
    ),
    "thunder_badge": _ObjectSelection(
        map_id=_MAP_VERMILION_GYM,
        object_index=0,  # VermilionGym.asm object_event #1: VERMILIONGYM_LT_SURGE
        rationale="This map's only OPP_LT_SURGE trainer record.",
    ),
    "rainbow_badge": _ObjectSelection(
        map_id=_MAP_CELADON_GYM,
        object_index=0,  # CeladonGym.asm object_event #1: CELADONGYM_ERIKA
        rationale="This map's only OPP_ERIKA trainer record.",
    ),
    "got_poke_flute": _ObjectSelection(
        map_id=_MAP_POKEMON_TOWER_7F,
        object_index=3,  # PokemonTower7F.asm object_event #4: POKEMONTOWER7F_MR_FUJI
        rationale="The only non-Rocket object on the floor; the three Rocket grunts precede him in source order.",
    ),
    "soul_badge": _ObjectSelection(
        map_id=_MAP_FUCHSIA_GYM,
        object_index=0,  # FuchsiaGym.asm object_event #1: FUCHSIAGYM_KOGA
        rationale="This map's only OPP_KOGA trainer record.",
    ),
    "marsh_badge": _ObjectSelection(
        map_id=_MAP_SAFFRON_GYM,
        object_index=0,  # SaffronGym.asm object_event #1: SAFFRONGYM_SABRINA
        rationale="This map's only OPP_SABRINA trainer record.",
    ),
    "volcano_badge": _ObjectSelection(
        map_id=_MAP_CINNABAR_GYM,
        object_index=0,  # CinnabarGym.asm object_event #1: CINNABARGYM_BLAINE
        rationale="This map's only OPP_BLAINE trainer record.",
    ),
    "earth_badge": _ObjectSelection(
        map_id=_MAP_VIRIDIAN_GYM,
        object_index=0,  # ViridianGym.asm object_event #1: VIRIDIANGYM_GIOVANNI
        rationale="This map's only OPP_GIOVANNI trainer record.",
    ),
    "beat_champion": _ObjectSelection(
        map_id=_MAP_CHAMPIONS_ROOM,
        object_index=0,  # ChampionsRoom.asm object_event #1: CHAMPIONSROOM_RIVAL
        rationale="The rival, not Oak (object index 1), is the final battle.",
    ),
}


def resolve_target_coordinates(
    rom_maps_by_id: dict[int, RomMap],
) -> dict[str, tuple[int, int]]:
    """Every milestone id's `(x, y)`, read off the selected object record in
    `rom_maps_by_id` (as returned by `rom_maps.parse_all_maps`)."""
    return {
        milestone_id: (
            rom_maps_by_id[selection.map_id].objects[selection.object_index].x,
            rom_maps_by_id[selection.map_id].objects[selection.object_index].y,
        )
        for milestone_id, selection in MILESTONE_OBJECT_SELECTIONS.items()
    }
