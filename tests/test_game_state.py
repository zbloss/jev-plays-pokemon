from pathlib import Path

import pytest
from pyboy import PyBoy

from jev_plays_pokemon.game_state import (
    BattleResultTracker,
    BattleState,
    extract_game_state,
)

ROM_PATH = Path(__file__).resolve().parent.parent / "pokemon_red.gb"

# `pokemon_red.gb` is gitignored (redistributing a copyrighted ROM isn't
# appropriate for a public repo/CI runner), so it only exists in local dev
# checkouts that added their own copy. These tests run for real against
# PyBoy + the ROM wherever it's present, and skip (rather than fail) where
# it's not.
pytestmark = pytest.mark.skipif(
    not ROM_PATH.exists(), reason=f"{ROM_PATH} not present locally"
)


@pytest.fixture
def pyboy():
    instance = PyBoy(str(ROM_PATH), window="null")
    instance.set_emulation_speed(0)
    instance.tick(1, False)
    yield instance
    instance.stop(save=False)


def test_fresh_boot_reads_empty_baseline_state(pyboy):
    state = extract_game_state(pyboy)

    assert state.party == ()
    assert state.money == 0
    assert state.inventory == ()
    assert state.badges == ()
    assert state.event_flags == frozenset()
    assert state.battle == BattleState(
        in_battle=False, battle_type="none", opponent_species=None, opponent_level=None
    )
    assert state.dialog_open is False


def test_party_is_read_and_decoded_from_the_real_party_struct_layout(pyboy):
    pyboy.memory[0xD163] = 2
    slot_0 = 0xD16B
    slot_1 = slot_0 + 0x2C

    # Slot 0: CHARMANDER (0xB0), level 5, 18/19 HP, poisoned, one known move w/ PP.
    pyboy.memory[slot_0 + 0] = 0xB0
    pyboy.memory[slot_0 + 1] = 0x00
    pyboy.memory[slot_0 + 2] = 0x12  # HP = 18
    pyboy.memory[slot_0 + 4] = 0b0000_1000  # poisoned
    pyboy.memory[slot_0 + 8] = 0x21  # SCRATCH (first move)
    pyboy.memory[slot_0 + 9] = 0
    pyboy.memory[slot_0 + 10] = 0
    pyboy.memory[slot_0 + 11] = 0
    pyboy.memory[slot_0 + 29] = 0b1100_0101  # PP Up bonus bits set + PP = 5
    pyboy.memory[slot_0 + 33] = 5  # level
    pyboy.memory[slot_0 + 34] = 0x00
    pyboy.memory[slot_0 + 35] = 0x13  # max HP = 19

    # Slot 1: SQUIRTLE (0xB1), level 10, full HP, no status.
    pyboy.memory[slot_1 + 0] = 0xB1
    pyboy.memory[slot_1 + 1] = 0x00
    pyboy.memory[slot_1 + 2] = 0x1E  # HP = 30
    pyboy.memory[slot_1 + 4] = 0
    pyboy.memory[slot_1 + 33] = 10
    pyboy.memory[slot_1 + 34] = 0x00
    pyboy.memory[slot_1 + 35] = 0x1E  # max HP = 30

    state = extract_game_state(pyboy)

    assert len(state.party) == 2
    charmander = state.party[0]
    assert charmander.species == "CHARMANDER"
    assert charmander.level == 5
    assert charmander.hp == 18
    assert charmander.max_hp == 19
    assert charmander.status == "POISONED"
    assert charmander.moves == (0x21, 0, 0, 0)
    assert charmander.pp == (0b0000_0101, 0, 0, 0)

    squirtle = state.party[1]
    assert squirtle.species == "SQUIRTLE"
    assert squirtle.level == 10
    assert squirtle.hp == 30
    assert squirtle.max_hp == 30
    assert squirtle.status == "OK"


def test_sleep_status_takes_priority_over_other_status_bits(pyboy):
    pyboy.memory[0xD163] = 1
    slot_0 = 0xD16B
    pyboy.memory[slot_0 + 0] = 0xB0
    pyboy.memory[slot_0 + 4] = 0b0000_0011  # 3 turns asleep

    state = extract_game_state(pyboy)

    assert state.party[0].status == "ASLEEP"


def test_money_is_decoded_from_packed_bcd(pyboy):
    pyboy.memory[0xD347] = 0x12
    pyboy.memory[0xD348] = 0x34
    pyboy.memory[0xD349] = 0x56

    state = extract_game_state(pyboy)

    assert state.money == 123456


def test_badges_are_resolved_to_names_in_bit_order(pyboy):
    pyboy.memory[0xD356] = 0b0000_0101  # boulder + thunder

    state = extract_game_state(pyboy)

    assert state.badges == ("BOULDERBADGE", "THUNDERBADGE")


def test_inventory_is_read_until_the_bag_sentinel(pyboy):
    pyboy.memory[0xD31D] = 2
    pyboy.memory[0xD31E] = 0x14  # POTION
    pyboy.memory[0xD31F] = 5
    pyboy.memory[0xD320] = 0x01  # MASTER BALL
    pyboy.memory[0xD321] = 1
    pyboy.memory[0xD322] = 0xFF  # sentinel

    state = extract_game_state(pyboy)

    assert [(item.item, item.quantity) for item in state.inventory] == [
        ("POTION", 5),
        ("MASTER BALL", 1),
    ]


def test_event_flags_reports_set_bit_indices(pyboy):
    pyboy.memory[0xD747] = 0b0000_0001  # bit 0 -> flag index 0
    pyboy.memory[0xD747 + 1] = 0b0000_0010  # bit 1 of byte 1 -> flag index 9

    state = extract_game_state(pyboy)

    assert state.event_flags == frozenset({0, 9})


def test_wild_battle_reports_opponent_species_and_level(pyboy):
    pyboy.memory[0xD057] = 1  # wild battle
    pyboy.memory[0xD059] = 0x54  # PIKACHU
    pyboy.memory[0xD127] = 7

    state = extract_game_state(pyboy)

    assert state.battle.in_battle is True
    assert state.battle.battle_type == "wild"
    assert state.battle.opponent_species == "PIKACHU"
    assert state.battle.opponent_level == 7


def test_trainer_battle_reports_level_but_not_species(pyboy):
    pyboy.memory[0xD057] = 2  # trainer battle
    pyboy.memory[0xD127] = 12

    state = extract_game_state(pyboy)

    assert state.battle.battle_type == "trainer"
    assert state.battle.in_battle is True
    assert state.battle.opponent_species is None
    assert state.battle.opponent_level == 12


def test_lost_battle_is_reported_as_not_in_battle(pyboy):
    pyboy.memory[0xD057] = 0xFF

    state = extract_game_state(pyboy)

    assert state.battle.battle_type == "lost"
    assert state.battle.in_battle is False


def test_last_battle_result_is_none_without_a_tracker(pyboy):
    pyboy.memory[0xD057] = 1  # wild battle
    pyboy.memory[0xCF0B] = 0x00  # win, but there's no tracker to latch it

    state = extract_game_state(pyboy)

    assert state.last_battle_result is None


def test_last_battle_result_is_none_before_any_battle_has_ended(pyboy):
    tracker = BattleResultTracker()

    state = extract_game_state(pyboy, battle_result_tracker=tracker)

    assert state.last_battle_result is None


def test_last_battle_result_latches_a_win_on_the_nonzero_to_zero_transition(pyboy):
    tracker = BattleResultTracker()

    pyboy.memory[0xD057] = 1  # wild battle in progress
    state = extract_game_state(pyboy, battle_result_tracker=tracker)
    assert state.last_battle_result is None

    pyboy.memory[0xD057] = 0  # battle just ended
    pyboy.memory[0xCF0B] = 0x00  # win
    state = extract_game_state(pyboy, battle_result_tracker=tracker)

    assert state.last_battle_result == "win"


def test_last_battle_result_latches_a_loss_from_the_lost_battle_state(pyboy):
    tracker = BattleResultTracker()

    pyboy.memory[0xD057] = 2  # trainer battle in progress
    extract_game_state(pyboy, battle_result_tracker=tracker)

    pyboy.memory[0xD057] = 0xFF  # lost
    extract_game_state(pyboy, battle_result_tracker=tracker)

    pyboy.memory[0xD057] = 0  # settles back to no battle
    pyboy.memory[0xCF0B] = 0x01  # lose
    state = extract_game_state(pyboy, battle_result_tracker=tracker)

    assert state.last_battle_result == "lose"


def test_last_battle_result_latches_a_draw(pyboy):
    tracker = BattleResultTracker()

    pyboy.memory[0xD057] = 2  # trainer battle in progress
    extract_game_state(pyboy, battle_result_tracker=tracker)

    pyboy.memory[0xD057] = 0
    pyboy.memory[0xCF0B] = 0x02  # draw
    state = extract_game_state(pyboy, battle_result_tracker=tracker)

    assert state.last_battle_result == "draw"


def test_last_battle_result_stays_latched_until_the_next_battle_ends(pyboy):
    tracker = BattleResultTracker()

    pyboy.memory[0xD057] = 1
    extract_game_state(pyboy, battle_result_tracker=tracker)
    pyboy.memory[0xD057] = 0
    pyboy.memory[0xCF0B] = 0x00  # win
    extract_game_state(pyboy, battle_result_tracker=tracker)

    # `wBattleResult` gets reused/cleared between battles - reading it on a
    # tick that isn't the latch transition must not affect the exposed value.
    pyboy.memory[0xCF0B] = 0xFF
    state = extract_game_state(pyboy, battle_result_tracker=tracker)

    assert state.last_battle_result == "win"


def test_dialog_open_is_detected_from_the_continue_arrow_tile(pyboy):
    lcdc = pyboy.memory[0xFF40]
    window_select_bit = (lcdc >> 6) & 1
    tiledata_select_bit = (lcdc >> 4) & 1
    base = 0x9C00 if window_select_bit else 0x9800
    address = base + 32 * 16 + 18
    raw_byte = 238 if tiledata_select_bit else ((238 - 256 + 128) ^ 0x80) & 0xFF

    assert pyboy.tilemap_window[18, 16] != 238
    assert extract_game_state(pyboy).dialog_open is False

    pyboy.memory[address] = raw_byte

    assert pyboy.tilemap_window[18, 16] == 238
    assert extract_game_state(pyboy).dialog_open is True


def test_map_id_and_player_position_are_read_directly(pyboy):
    pyboy.memory[0xD35E] = 0x0C  # route_1
    pyboy.memory[0xD362] = 5  # x
    pyboy.memory[0xD361] = 7  # y

    state = extract_game_state(pyboy)

    assert state.map_id == 0x0C
    assert state.map_name == "Route 1"
    assert state.player_x == 5
    assert state.player_y == 7
