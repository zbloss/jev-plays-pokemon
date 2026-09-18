"""Item ID -> display name lookup.

Sourced from `pret/pokered`, the canonical Pokemon Red/Blue disassembly:
- IDs: `constants/item_constants.asm`
- Display text for items $01-$61: `data/items/names.asm`

TM/HM item IDs ($C4-$FA) aren't in that display-name table; the game
renders them as "HM01".."HM05" / "TM01".."TM50" by their position among the
`add_hm`/`add_tm` definitions in `item_constants.asm` (HMs start at $C4,
TMs at $C9), which is reproduced directly below rather than hardcoding each
entry.
"""

_ITEM_NAMES: dict[int, str] = {
    0x01: "MASTER BALL",
    0x02: "ULTRA BALL",
    0x03: "GREAT BALL",
    0x04: "POKé BALL",
    0x05: "TOWN MAP",
    0x06: "BICYCLE",
    0x07: "?????",  # SURFBOARD - unused, pokered renders it as "?????"
    0x08: "SAFARI BALL",
    0x09: "POKéDEX",
    0x0A: "MOON STONE",
    0x0B: "ANTIDOTE",
    0x0C: "BURN HEAL",
    0x0D: "ICE HEAL",
    0x0E: "AWAKENING",
    0x0F: "PARLYZ HEAL",
    0x10: "FULL RESTORE",
    0x11: "MAX POTION",
    0x12: "HYPER POTION",
    0x13: "SUPER POTION",
    0x14: "POTION",
    0x15: "BOULDERBADGE",
    0x16: "CASCADEBADGE",
    0x17: "THUNDERBADGE",
    0x18: "RAINBOWBADGE",
    0x19: "SOULBADGE",
    0x1A: "MARSHBADGE",
    0x1B: "VOLCANOBADGE",
    0x1C: "EARTHBADGE",
    0x1D: "ESCAPE ROPE",
    0x1E: "REPEL",
    0x1F: "OLD AMBER",
    0x20: "FIRE STONE",
    0x21: "THUNDERSTONE",
    0x22: "WATER STONE",
    0x23: "HP UP",
    0x24: "PROTEIN",
    0x25: "IRON",
    0x26: "CARBOS",
    0x27: "CALCIUM",
    0x28: "RARE CANDY",
    0x29: "DOME FOSSIL",
    0x2A: "HELIX FOSSIL",
    0x2B: "SECRET KEY",
    0x2C: "?????",  # ITEM_2C - unused, pokered renders it as "?????"
    0x2D: "BIKE VOUCHER",
    0x2E: "X ACCURACY",
    0x2F: "LEAF STONE",
    0x30: "CARD KEY",
    0x31: "NUGGET",
    0x32: "PP UP",
    0x33: "POKé DOLL",
    0x34: "FULL HEAL",
    0x35: "REVIVE",
    0x36: "MAX REVIVE",
    0x37: "GUARD SPEC.",
    0x38: "SUPER REPEL",
    0x39: "MAX REPEL",
    0x3A: "DIRE HIT",
    0x3B: "COIN",
    0x3C: "FRESH WATER",
    0x3D: "SODA POP",
    0x3E: "LEMONADE",
    0x3F: "S.S.TICKET",
    0x40: "GOLD TEETH",
    0x41: "X ATTACK",
    0x42: "X DEFEND",
    0x43: "X SPEED",
    0x44: "X SPECIAL",
    0x45: "COIN CASE",
    0x46: "OAK's PARCEL",
    0x47: "ITEMFINDER",
    0x48: "SILPH SCOPE",
    0x49: "POKé FLUTE",
    0x4A: "LIFT KEY",
    0x4B: "EXP.ALL",
    0x4C: "OLD ROD",
    0x4D: "GOOD ROD",
    0x4E: "SUPER ROD",
    0x4F: "PP UP",
    0x50: "ETHER",
    0x51: "MAX ETHER",
    0x52: "ELIXER",
    0x53: "MAX ELIXER",
    0x54: "B2F",
    0x55: "B1F",
    0x56: "1F",
    0x57: "2F",
    0x58: "3F",
    0x59: "4F",
    0x5A: "5F",
    0x5B: "6F",
    0x5C: "7F",
    0x5D: "8F",
    0x5E: "9F",
    0x5F: "10F",
    0x60: "11F",
    0x61: "B4F",
    **{0xC4 + i: f"HM{i + 1:02d}" for i in range(5)},
    **{0xC9 + i: f"TM{i + 1:02d}" for i in range(50)},
}


def item_name(item_id: int) -> str:
    """Resolve a Pokemon Red item ID to its display name.

    Unknown/out-of-range IDs return a placeholder string instead of
    raising, so callers building game-state text never crash on a bad ID.
    """
    return _ITEM_NAMES.get(item_id, f"Unknown Item ({item_id})")
