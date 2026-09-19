"""Static Pokemon Red ID -> display name lookup tables.

Covers the RAM-exposed ID spaces that each have a single, fixed, documented
enum in `pret/pokered`'s disassembly constants: items (`item_name`), species
(`species_name`), map locations (`map_name`), and moves (`move_name`).

Menu option names are deliberately not included here. RAM only exposes a
menu's cursor *index* (`wCurrentMenuItem` et al. in `pret/pokered`'s
`ram/wram.asm`), not one shared ID space the way items/species/maps each
have one - every menu screen (main menu, battle menu, bag, ...) has its own
fixed option list keyed by that screen, not by a global ID. Per
`docs/research/pokemon-red-ram-map.md`'s recommendation, that's better
resolved as small, per-screen option lists hardcoded in whatever module
actually reads the menu-screen type and cursor index (the game-state
extraction module), not as a generic ID lookup table here.
"""

from jev_plays_pokemon.lookup.items import item_name
from jev_plays_pokemon.lookup.maps import map_name
from jev_plays_pokemon.lookup.moves import move_name
from jev_plays_pokemon.lookup.species import species_name

__all__ = ["item_name", "map_name", "move_name", "species_name"]
