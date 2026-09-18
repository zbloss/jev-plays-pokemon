# Research: Pokemon Red RAM-map / state-extraction resources

Resolves issue [#7](https://github.com/zbloss/jev-plays-pokemon/issues/7), child of [#1](https://github.com/zbloss/jev-plays-pokemon/issues/1).

## Question

Survey what's available for extracting structured game state from Pokemon Red's
emulator RAM (e.g. via PyBoy + known disassembly/RAM-map projects like `pokered`):
what's already solved vs. needs to be built, and which state fields (player
position, party, battle state, menus, dialog, etc.) are reliably readable this
way vs. likely needing the vision fallback.

## Top-line answer

Reading structured Pokemon Red state from Game Boy RAM is a **solved problem**
for nearly everything except on-screen text content. Three independent,
well-established primary sources agree on the same addresses, and one of them
(PyBoy itself) now ships a maintained, high-level Python API purpose-built for
Pokemon Red/Blue — we would not be reverse-engineering addresses from scratch.

- **Reliably RAM-readable:** player map ID + X/Y position, full party data
  (species, level, current/max HP, stats, status, moves, PP, trainer info),
  money, badges, bag/PC inventory, battle state (in-battle flag, battle type,
  opponent species/level, selected moves, battle result), event flags
  (story/quest progress), and even *whether a dialog box is currently open*
  (a specific tile in the background tilemap).
- **Not reliably RAM-readable (vision fallback territory):** the literal
  *text content* of dialog boxes and menus (RAM/tilemap tells you a box is
  open and where the cursor is, but decoding it requires either a
  ROM-text-table decoder — non-trivial, character-encoding-specific — or an
  OCR/vision read of the rendered screen), and any UI state that is purely
  visual/animated (e.g. mid-transition screen wipes, sprite animations).

## Sources consulted (primary)

1. **`pret/pokered`** — canonical disassembly of Pokemon Red/Blue.
   `ram/wram.asm`: https://github.com/pret/pokered/blob/master/ram/wram.asm
2. **PyBoy** — `Baekalfen/PyBoy`, the Python Game Boy emulator this project
   will use.
   - Core memory API docs: https://docs.pyboy.dk/
   - Game wrapper for Pokemon Red/Blue (source):
     https://github.com/Baekalfen/PyBoy/blob/master/pyboy/plugins/game_wrapper_pokemon_gen1.py
   - Game wrapper docs: https://docs.pyboy.dk/plugins/game_wrapper_pokemon_gen1.html
   - Plugin directory (confirms which games have dedicated wrappers):
     https://github.com/Baekalfen/PyBoy/tree/master/pyboy/plugins
3. **`PWhiddy/PokemonRedExperiments`** — widely-used Pokemon Red RL project;
   its RAM addresses are independently verified against the real compiled ROM
   (not just the disassembly's symbolic layout).
   - `baselines/red_gym_env.py`: https://github.com/PWhiddy/PokemonRedExperiments/blob/master/baselines/red_gym_env.py
   - `baselines/memory_addresses.py`: https://github.com/PWhiddy/PokemonRedExperiments/blob/master/baselines/memory_addresses.py
4. **Data Crystal RAM map** — community disassembly-derived reference wiki,
   cross-checked against the above (used only to corroborate, not as a
   standalone source): https://datacrystal.tcrf.net/wiki/Pok%C3%A9mon_Red_and_Blue/RAM_map
5. **`drubinstein/pokerl` ("Pokémon RL") docs** — a from-scratch bot-building
   writeup that explicitly documents which pokered WRAM symbols it read
   directly vs. where it had to hook ROM code because RAM alone wasn't
   enough (e.g. detecting a successful `CUT`):
   https://drubinstein.github.io/pokerl/docs/chapter-3/reading-asm/

## What's already solved: PyBoy's native Pokemon Gen 1 game wrapper

The most important finding is that **PyBoy already ships a dedicated,
maintained game wrapper for Pokemon Red/Blue**, at
`pyboy/plugins/game_wrapper_pokemon_gen1.py`. Its class docstring states:
"This class wraps Pokemon Red/Blue, and provides basic access for AIs."
(source: raw file linked above). This is a materially better starting point
than hand-rolling address reads, because it's shipped, tested, and versioned
with the emulator we're already using.

`enabled()` detects the wrapper by cartridge title and explicitly matches
only `"POKEMON RED"` and `"POKEMON BLUE"` — Yellow is **not** recognized by
this wrapper (source: `game_wrapper_pokemon_gen1.py`). Since this project
targets Pokemon Red specifically, that's a non-issue, but it should be
verified again once issue #6's ROM-version audit lands.

It exposes, as high-level Python properties/methods (source:
`game_wrapper_pokemon_gen1.py` + https://docs.pyboy.dk/plugins/game_wrapper_pokemon_gen1.html):

- `party` — list of dicts, one per party Pokemon, each with `species`,
  `level`, `hp`, `max_hp`, `attack`, `defense`, `speed`, `special`, `status`,
  `type1`, `type2`, `moves` (4), `pp`, `experience`, DVs, `catch_rate`,
  `ot_name`, `nickname`, `ot_id`.
- `money` — integer 0–999,999.
- `inventory` — list of `{item, quantity}` dicts (bag contents).
- Mutators for training/testing use (`set_party`, `add_pokemon`,
  `set_money`, `set_inventory`, `set_badge`, `set_event_flag`, `warp`,
  `start_wild_battle`, `start_trainer_battle`) — useful for our own test
  fixtures/harness, not just reads.
- `game_area_collision()` / `_get_screen_walkable_matrix()` — a walkable-tile
  matrix for the current screen, derived from the background tilemap.
- Battle-adjacent addresses used internally: `CURRENT_OPPONENT_ADDRESS`,
  `CURRENT_ENEMY_LEVEL_ADDRESS`, `BATTLE_TYPE_ADDRESS`, `TRAINER_NUMBER_ADDRESS`,
  `WARP_DESTINATION_MAP_ADDRESS`.
- Badges/events: `OBTAINED_BADGES_ADDRESS`, `EVENT_FLAGS_ADDRESS`, with
  `set_badge`/`reset_badge`/`set_event_flag`/`reset_event_flag` taking either
  a name or bit/id.

Notably, **the wrapper does not expose player X/Y position or current map ID**
as a property (confirmed by searching the raw source — no position/map
accessors present). Those still need to be read as raw addresses via
`pyboy.memory[...]` (see next section) — this is a small gap to fill
ourselves, not a blocker.

It also has a documented, RAM/tilemap-based way to detect "is a dialog box
currently showing," used internally by its own `_skip_dialogue()` helper:
it polls `self.pyboy.tilemap_window[18, 16] != 238` — tile `238` at screen
tile position (18, 16) is literally the dialog "continue" arrow glyph
(source: `game_wrapper_pokemon_gen1.py`, method `_skip_dialogue`). This means
"a dialog box is open and waiting for input" is RAM/tilemap-detectable even
though the box's *text content* is not — exactly the boundary this ticket
was asked to establish.

PyBoy's raw memory API underneath all of this is simple and already fits our
needs: `pyboy.memory[0xD362]` reads a single byte, `pyboy.memory[0xC000:0xC010]`
reads a range, and `pyboy.memory[bank, addr]` reads a specific ROM/RAM bank
(source: https://docs.pyboy.dk/). Anything the game wrapper doesn't expose,
we can read directly with these primitives using addresses below.

## Raw addresses (Pokemon Red, English ROM) for fields the wrapper doesn't expose

These are the addresses `PokemonRedExperiments` uses against the real
compiled ROM (source: `memory_addresses.py`, linked above), cross-checked
against the community Data Crystal RAM map (linked above) and the symbolic
names in `pret/pokered`'s `ram/wram.asm` (linked above):

| Field | Address | Symbol (pokered) | Source |
|---|---|---|---|
| Current map ID | `0xD35E` | `wCurMap` | PokemonRedExperiments `MAP_N_ADDRESS`; pokered `wram.asm` |
| Player Y position | `0xD361` | `wYCoord` | PokemonRedExperiments `Y_POS_ADDRESS`; Data Crystal |
| Player X position | `0xD362` | `wXCoord` | PokemonRedExperiments `X_POS_ADDRESS`; Data Crystal |
| Party size | `0xD163` | `wPartyCount` | PokemonRedExperiments `PARTY_SIZE_ADDRESS` |
| Money (3-byte BCD) | `0xD347`–`0xD349` | `wPlayerMoney` | PokemonRedExperiments `MONEY_ADDRESS_1/2/3`; Data Crystal |
| Badges (bitfield) | `0xD356` | `wObtainedBadges` | PokemonRedExperiments `BADGE_COUNT_ADDRESS`; Data Crystal |
| Event flags (story/quest progress) | `0xD747`–`0xD886` | `wEventFlags` | PokemonRedExperiments `EVENT_FLAGS_START/END_ADDRESS` |
| In-battle flag | ~`0xD057` | `wIsInBattle` | Data Crystal (`wram.asm` confirms symbol exists; exact offset not independently verified against compiled ROM in this pass) |
| Party Pokemon 1 species/HP/level block | starts `0xD16B`, 44 bytes/slot (`party_struct`) | `wPartyMon1`..`wPartyMon6` | Data Crystal; pokered `wram.asm` struct layout — but prefer the PyBoy wrapper's parsed `party` property over hand-parsing this struct |
| Enemy party count (in battle) | `0xD89C` | — | PokemonRedExperiments `red_gym_env.py` (`enemy_poke_count`) |
| Opponent level (in battle) | `0xCFF3` | — | PokemonRedExperiments `red_gym_env.py` (`opponent_level`) |

**Caveat on precision:** the `pret/pokered` disassembly defines WRAM layout
*symbolically* (via `rsset`/struct macros in `ram/wram.asm`), so it doesn't
print literal hex addresses directly in the source — the numeric offsets
above come from `PokemonRedExperiments`, whose addresses are exercised
against the real, running, compiled ROM via PyBoy and are the values we'd
actually want to hardcode. Data Crystal's numbers agree with them wherever
compared. Before relying on any address not in the table above, cross-check
it in at least two of these three sources — Data Crystal in particular is a
community wiki and should not be trusted standalone (per the ticket's
guidance).

## Party Pokemon struct detail (if we ever need to hand-parse instead of using PyBoy's `party` property)

Per `pret/pokered`'s `ram/wram.asm` `party_struct` macro and corroborated by
Data Crystal: each party slot is a fixed-size struct (43–44 bytes depending
on how OT-name/nickname are counted) containing species, current HP,
status condition, type1/type2, catch rate/held item byte, move IDs (4),
trainer ID, experience (3 bytes), effort values (HP/Atk/Def/Spd/Special),
individual values (packed nibbles), PP per move (4), and level — followed by
separate parallel arrays for OT name and nickname strings. **We should not
need to hand-parse this**: PyBoy's `game_wrapper_pokemon_gen1.py` already
does this parsing and returns clean dicts via its `party` property, which is
the API this project should call.

## Battle state

Reliably readable via a mix of the PyBoy wrapper's exposed addresses and raw
reads: whether a battle is active, whether it's wild vs. trainer, opponent
species/level, party Pokemon's live HP/status during battle (same `party`
data, refreshed per frame), and battle outcome. Per-turn *menu selection*
(which of Fight/Bag/Pokemon/Run is highlighted) is also RAM-backed —
`pret/pokered`'s `wram.asm` defines `wCurrentMenuItem`, `wTopMenuItemX/Y`,
`wMaxMenuItem`, `wMenuItemToSwap` for general menu cursor state (source:
`ram/wram.asm`), and battle-specific `wPlayerSelectedMove` /
`wEnemySelectedMove` / `wMoveMenuType` for move-selection state (same
source). So "what menu option is currently highlighted" is RAM-readable;
"what the menu options say as rendered text" is not (they're static per
screen type, though, so for known menu types this can be hardcoded rather
than read at all).

## What is NOT reliably RAM-readable → vision fallback candidates

1. **Dialog/NPC text content.** RAM tells us a text box is open (tile `238`
   check above) and, via `pret/pokered`, could in principle be decoded by
   walking the game's custom character-encoding text buffer in RAM/VRAM —
   but this is nontrivial (Pokemon Gen 1 uses a proprietary text encoding,
   not ASCII) and neither PyBoy's wrapper nor PokemonRedExperiments attempt
   it. `drubinstein/pokerl`'s writeup (source linked above) explicitly frames
   dialogue/menu *text* as outside what it reads from RAM, resorting to
   ROM-code hooks only for specific mechanical checks (e.g. "was Cut just
   used successfully"), not for reading prose. This is the strongest
   candidate for the vision-fallback tool call.
2. **Rendered menu/list text in general** (e.g. the literal item names in a
   Bag menu, PC box labels) — the *selection index* is RAM-readable, but
   Jev would need either a hardcoded item-ID→name table (fully solvable
   without vision, since item IDs are a fixed enum defined in `pokered`) or
   a vision read if we don't want to maintain that table ourselves. This is
   a "build it" case, not a "needs vision" case, since the enums are static
   and documented in `pret/pokered`'s constants files.
3. **Pure animation/visual transition states** (screen wipes, sprite
   movement mid-frame, battle animations) — there's no clean RAM flag for
   "is a fade transition visually in progress" in general; these are
   probably best handled by rate-limiting/debouncing state reads rather than
   either RAM or vision (i.e. wait for the next stable frame).

## Recommendation for the MVP architecture

- Use PyBoy's `game_wrapper_pokemon_gen1.py` (`pyboy.game_wrapper`) as the
  primary state source for party, money, inventory, badges, event flags, and
  dialog-open detection — it's already built and shipped with our emulator
  dependency.
- Supplement with direct `pyboy.memory[...]` reads for player map ID/X/Y
  position and any battle-specific fields not exposed by the wrapper
  property surface, using the addresses table above (re-verify each address
  against a running emulator instance before relying on it in production,
  since none of these were independently re-verified against our own ROM
  copy in this research pass — this was a documentation survey, not a
  build/verify step, per this ticket's scope).
- Build a static item-ID/map-ID/species-ID → name lookup table from
  `pret/pokered`'s constants files (e.g. `constants/item_constants.asm`,
  `constants/map_constants.asm`) rather than reading names from the screen.
- Reserve the vision-model fallback specifically for dialog/NPC text content
  (and any other on-screen prose), since that's the one category with no
  practical RAM-only path.

## Open follow-ups for a later session

- Independently verify the addresses in this doc against a running PyBoy +
  Pokemon Red instance (this ticket was documentation-only, per its scope).
- Confirm exact byte offset/verification for `wIsInBattle` and the full
  `party_struct` byte layout against the compiled ROM, not just the
  disassembly source and secondary wikis.
- Decide whether to build the item/map/species ID→name tables from
  `pret/pokered` constants now, or defer to whenever the vision-fallback
  design (issue chain under #1) is scoped.
