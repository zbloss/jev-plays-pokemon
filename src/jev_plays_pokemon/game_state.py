"""RAM-based Pokemon Red game-state extraction.

Reads PyBoy's emulator memory each turn and assembles it into a single
structured `GameState`. Every field here is RAM-sourced, never vision-sourced
(see `CONTEXT.md`'s "Game state" entry) - this module owns all RAM-sourced
fields and never decides per turn whether to fall back to vision.

Addresses are sourced from `pret/pokered`'s disassembly (`ram/wram.asm`,
`macros/ram.asm`'s `party_struct`/`box_struct` macros, `constants/`), cross-
checked against `PWhiddy/PokemonRedExperiments`'s `baselines/memory_addresses.py`
(independently verified against a compiled ROM) and the Data Crystal RAM map,
then re-verified by booting this repo's own `pokemon_red.gb` via PyBoy and
reading them directly - per this ticket's requirement, since
`docs/research/pokemon-red-ram-map.md`'s address table was a documentation
survey, not a build/verify pass.

PyBoy 2.2.0 (the version this project depends on) does *not* ship the rich
`party`/`money`/`inventory`/badge-mutator API that `docs/research/pokemon-red-ram-map.md`
describes - that API only exists on PyBoy's unreleased development branch, not
in any published version. `pyboy.game_wrapper` for Pokemon Red/Blue in 2.2.0
exposes only `game_area`/`game_area_collision`/`start_game`/`reset_game`
(confirmed by reading the installed package's source). Every field below is
therefore read directly via `pyboy.memory[...]` / `pyboy.tilemap_window[...]`,
PyBoy's stable low-level APIs, rather than via wrapper properties that don't
exist at this pinned version.
"""

from dataclasses import dataclass
from typing import Literal

from pyboy import PyBoy

from jev_plays_pokemon.lookup import item_name, map_name, species_name

_PARTY_COUNT_ADDRESS = 0xD163
_PARTY_MON_MAX = 6
_PARTY_MON_BASE_ADDRESS = 0xD16B
# `PARTYMON_STRUCT_LENGTH` in pokered's `constants/pokemon_data_constants.asm`
_PARTY_MON_SIZE = 0x2C

# Byte offsets within one party Pokemon struct, per pokered's `party_struct`/
# `box_struct` macros (`macros/ram.asm`).
_MON_OFFSET_SPECIES = 0
_MON_OFFSET_HP = 1  # 2 bytes, big-endian
_MON_OFFSET_STATUS = 4
_MON_OFFSET_MOVES = 8  # 4 bytes, one per move slot; 0 = empty slot
_MON_OFFSET_PP = 29  # 4 bytes; low 6 bits = current PP, top 2 bits = PP Up count
_MON_OFFSET_LEVEL = 33
_MON_OFFSET_MAX_HP = 34  # 2 bytes, big-endian

# Gen 1 non-volatile status bitmask, per pokered's `constants/battle_constants.asm`.
_STATUS_SLEEP_MASK = 0b0000_0111
_STATUS_POISONED_BIT = 1 << 3
_STATUS_BURNED_BIT = 1 << 4
_STATUS_FROZEN_BIT = 1 << 5
_STATUS_PARALYZED_BIT = 1 << 6

_MONEY_ADDRESS = 0xD347  # 3 bytes, packed BCD, most-significant byte first

_BADGES_ADDRESS = 0xD356
# Badge bit position -> its item ID (items.py already has BOULDERBADGE..EARTHBADGE
# as a contiguous run starting at 0x15, in the same bit order pokered uses).
_BADGE_ITEM_IDS = (0x15, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C)

_BAG_ITEM_COUNT_ADDRESS = 0xD31D
_BAG_ITEMS_ADDRESS = 0xD31E
_BAG_ITEM_CAPACITY = 20
_BAG_SENTINEL = 0xFF

_EVENT_FLAGS_START_ADDRESS = 0xD747
_EVENT_FLAGS_END_ADDRESS = 0xD886

# `wIsInBattle`: 0 = no battle, 1 = wild battle, 2 = trainer battle, -1 (0xFF) =
# lost battle (pokered's own comment on the symbol).
_IS_IN_BATTLE_ADDRESS = 0xD057
_BATTLE_TYPE_LOST = 0xFF
_BATTLE_TYPE_WILD = 1
_BATTLE_TYPE_TRAINER = 2
_CURRENT_OPPONENT_ADDRESS = 0xD059
_CURRENT_ENEMY_LEVEL_ADDRESS = 0xD127

# `wBattleResult` (#75): a plain, standalone `db` in pokered's `ram/wram.asm`
# ("WRAM" section, right after `wBoughtOrSoldItemInMart` and right before
# `wAutoTextBoxDrawingControl`, outside every `UNION`/`ENDU` block) - not
# inside a union as previously assumed here. Its address is computed by
# summing every preceding struct/db/ds/UNION size from WRAM0's origin per
# pokered's own `layout.link` (WRAM0 starts its packed layout at $c100, not
# $c000 - the "Audio RAM" section ahead of it is placed separately), the same
# method this file already uses for its other addresses, and cross-checked
# by reproducing that sum in full for `wIsInBattle`/`wCurOpponent`/
# `wPartyCount`/`wNumBagItems`/`wPlayerMoney`/`wObtainedBadges`/`wCurMap`/
# `wYCoord`/`wXCoord` - every other address already verified in this file -
# and getting an exact match for all nine.
_BATTLE_RESULT_ADDRESS = 0xCF0B
_BATTLE_RESULT_WIN = 0x00
_BATTLE_RESULT_LOSE = 0x01
_BATTLE_RESULT_DRAW = 0x02

BattleResult = Literal["win", "lose", "draw"]

_MAP_ID_ADDRESS = 0xD35E
_PLAYER_Y_ADDRESS = 0xD361
_PLAYER_X_ADDRESS = 0xD362

# The dialog "continue" arrow glyph, at a fixed window-tilemap position,
# per `docs/research/pokemon-red-ram-map.md`'s citation of PyBoy's own
# `_skip_dialogue` helper. `pyboy.tilemap_window` is PyBoy's stable, public,
# game-wrapper-independent tilemap API (unlike the Gen1 wrapper's convenience
# methods, it does exist in 2.2.0).
_DIALOG_ARROW_TILE = 238
_DIALOG_ARROW_COLUMN = 18
_DIALOG_ARROW_ROW = 16


@dataclass(frozen=True)
class PartyPokemon:
    species: str
    level: int
    hp: int
    max_hp: int
    status: str
    moves: tuple[int, int, int, int]
    pp: tuple[int, int, int, int]


@dataclass(frozen=True)
class InventoryItem:
    item: str
    quantity: int


@dataclass(frozen=True)
class BattleState:
    in_battle: bool
    battle_type: str  # "none" | "wild" | "trainer" | "lost"
    opponent_species: str | None
    opponent_level: int | None


@dataclass(frozen=True)
class GameState:
    party: tuple[PartyPokemon, ...]
    money: int
    inventory: tuple[InventoryItem, ...]
    badges: tuple[str, ...]
    event_flags: frozenset[int]
    battle: BattleState
    dialog_open: bool
    map_id: int
    map_name: str
    player_x: int
    player_y: int
    last_battle_result: BattleResult | None = None


class BattleResultTracker:
    """Latches `wBattleResult` at the instant `wIsInBattle` transitions from
    nonzero (wild/trainer/lost) to `0` (#75) - the only instant its value
    means anything, since the game reuses/clears the byte between battles.

    One instance per run, fed every turn's `BattleState.battle_type` via
    `update`. Returns the most recently latched result, or `None` before any
    battle has ended.
    """

    def __init__(self) -> None:
        self._battle_in_progress = False
        self._latched: BattleResult | None = None

    def update(self, memory, battle_type: str) -> BattleResult | None:
        battle_in_progress = battle_type != "none"
        if self._battle_in_progress and not battle_in_progress:
            raw = memory[_BATTLE_RESULT_ADDRESS]
            if raw == _BATTLE_RESULT_WIN:
                self._latched = "win"
            elif raw == _BATTLE_RESULT_LOSE:
                self._latched = "lose"
            elif raw == _BATTLE_RESULT_DRAW:
                self._latched = "draw"
        self._battle_in_progress = battle_in_progress
        return self._latched


def extract_game_state(
    pyboy: PyBoy, *, battle_result_tracker: BattleResultTracker | None = None
) -> GameState:
    """Read one turn's structured `GameState` from a running PyBoy instance.

    `battle_result_tracker` is optional (#75): callers that don't pass one
    (most tests, `benchmark.py`, `navigation.py`) get `last_battle_result=None`
    on every call, the same as before this field existed. Production wiring
    (`main.py`) passes one long-lived tracker across turns so the latch
    survives from the turn a battle ends to whenever it's next read.
    """
    memory = pyboy.memory
    battle = _read_battle(memory)
    last_battle_result = (
        battle_result_tracker.update(memory, battle.battle_type)
        if battle_result_tracker is not None
        else None
    )

    return GameState(
        party=_read_party(memory),
        money=_read_money(memory),
        inventory=_read_inventory(memory),
        badges=_read_badges(memory),
        event_flags=_read_event_flags(memory),
        battle=battle,
        dialog_open=pyboy.tilemap_window[_DIALOG_ARROW_COLUMN, _DIALOG_ARROW_ROW]
        == _DIALOG_ARROW_TILE,
        map_id=memory[_MAP_ID_ADDRESS],
        map_name=map_name(memory[_MAP_ID_ADDRESS]),
        player_x=memory[_PLAYER_X_ADDRESS],
        player_y=memory[_PLAYER_Y_ADDRESS],
        last_battle_result=last_battle_result,
    )


def _read_u16_be(memory, address: int) -> int:
    return (memory[address] << 8) | memory[address + 1]


def _decode_status(status_byte: int) -> str:
    if status_byte & _STATUS_SLEEP_MASK:
        return "ASLEEP"
    if status_byte & _STATUS_POISONED_BIT:
        return "POISONED"
    if status_byte & _STATUS_BURNED_BIT:
        return "BURNED"
    if status_byte & _STATUS_FROZEN_BIT:
        return "FROZEN"
    if status_byte & _STATUS_PARALYZED_BIT:
        return "PARALYZED"
    return "OK"


def _read_party(memory) -> tuple[PartyPokemon, ...]:
    count = min(memory[_PARTY_COUNT_ADDRESS], _PARTY_MON_MAX)
    party = []
    for slot in range(count):
        base = _PARTY_MON_BASE_ADDRESS + slot * _PARTY_MON_SIZE
        party.append(
            PartyPokemon(
                species=species_name(memory[base + _MON_OFFSET_SPECIES]),
                level=memory[base + _MON_OFFSET_LEVEL],
                hp=_read_u16_be(memory, base + _MON_OFFSET_HP),
                max_hp=_read_u16_be(memory, base + _MON_OFFSET_MAX_HP),
                status=_decode_status(memory[base + _MON_OFFSET_STATUS]),
                moves=tuple(memory[base + _MON_OFFSET_MOVES + i] for i in range(4)),
                pp=tuple(memory[base + _MON_OFFSET_PP + i] & 0x3F for i in range(4)),
            )
        )
    return tuple(party)


def _read_money(memory) -> int:
    money = 0
    for offset in range(3):
        byte = memory[_MONEY_ADDRESS + offset]
        money = money * 100 + (byte >> 4) * 10 + (byte & 0x0F)
    return money


def _read_inventory(memory) -> tuple[InventoryItem, ...]:
    count = min(memory[_BAG_ITEM_COUNT_ADDRESS], _BAG_ITEM_CAPACITY)
    items = []
    for slot in range(count):
        item_id = memory[_BAG_ITEMS_ADDRESS + slot * 2]
        if item_id == _BAG_SENTINEL:
            break
        quantity = memory[_BAG_ITEMS_ADDRESS + slot * 2 + 1]
        items.append(InventoryItem(item=item_name(item_id), quantity=quantity))
    return tuple(items)


def _read_badges(memory) -> tuple[str, ...]:
    badge_byte = memory[_BADGES_ADDRESS]
    return tuple(
        item_name(item_id)
        for bit, item_id in enumerate(_BADGE_ITEM_IDS)
        if badge_byte & (1 << bit)
    )


def _read_event_flags(memory) -> frozenset[int]:
    flags = set()
    for offset in range(_EVENT_FLAGS_END_ADDRESS - _EVENT_FLAGS_START_ADDRESS):
        byte = memory[_EVENT_FLAGS_START_ADDRESS + offset]
        if not byte:
            continue
        for bit in range(8):
            if byte & (1 << bit):
                flags.add(offset * 8 + bit)
    return frozenset(flags)


def _read_battle(memory) -> BattleState:
    raw = memory[_IS_IN_BATTLE_ADDRESS]
    if raw == _BATTLE_TYPE_LOST:
        battle_type = "lost"
    elif raw == _BATTLE_TYPE_WILD:
        battle_type = "wild"
    elif raw == _BATTLE_TYPE_TRAINER:
        battle_type = "trainer"
    else:
        battle_type = "none"

    opponent_species = None
    opponent_level = None
    if battle_type == "wild":
        # In a wild battle, wCurOpponent holds the species ID directly.
        opponent_species = species_name(memory[_CURRENT_OPPONENT_ADDRESS])
        opponent_level = memory[_CURRENT_ENEMY_LEVEL_ADDRESS]
    elif battle_type == "trainer":
        # In a trainer battle, wCurOpponent holds the trainer class instead of
        # a species ID, so the active opponent's species isn't resolvable from
        # it - left unset rather than mislabeled.
        opponent_level = memory[_CURRENT_ENEMY_LEVEL_ADDRESS]

    return BattleState(
        in_battle=battle_type in ("wild", "trainer"),
        battle_type=battle_type,
        opponent_species=opponent_species,
        opponent_level=opponent_level,
    )
