"""Research smoke test for issue #8: one-off latency/shape check of a TypeSafe
Choice call, modeling "pick the next Game Boy button given battle state".

Reads TYPESAFE_API_KEY (and optionally TYPESAFE_ENV_FILE) from the environment /
a .env file via python-dotenv. Never hardcodes, logs, or prints the key value.

Usage:
    pip install typesafe-sdk python-dotenv
    # from the repo root, with TYPESAFE_API_KEY set in .env:
    python scripts/research/typesafe_smoke_test.py

    # or point at a specific .env file:
    TYPESAFE_ENV_FILE=/path/to/.env python scripts/research/typesafe_smoke_test.py
"""

import os
import time

from dotenv import load_dotenv

# Load .env if present; does not print or return the key.
_env_file = os.environ.get("TYPESAFE_ENV_FILE")
if _env_file:
    load_dotenv(_env_file)
else:
    load_dotenv()

if not os.environ.get("TYPESAFE_API_KEY"):
    raise SystemExit(
        "TYPESAFE_API_KEY not set in the environment (or TYPESAFE_ENV_FILE .env); "
        "skipping live smoke test."
    )

from typesafe_sdk import Choice, TypeSafeClient

STATE = {
    "screen": "battle",
    "player_pokemon": {"name": "CHARMANDER", "hp": 14, "max_hp": 19, "level": 8},
    "enemy_pokemon": {"name": "RATTATA", "hp": 6, "max_hp": 20, "level": 6},
    "menu": "FIGHT/PKMN/ITEM/RUN cursor on FIGHT",
    "available_moves": ["SCRATCH", "GROWL", "EMBER"],
}

QUESTIONS = {
    "next_button": Choice(
        instructions=(
            "Given the current Game Boy Pokemon battle state, which single button "
            "should be pressed next to make tactical progress?"
        ),
        criteria={
            "A": "Confirm/select the currently highlighted menu option",
            "B": "Cancel/back out of the current menu",
            "UP": "Move the menu cursor up",
            "DOWN": "Move the menu cursor down",
            "LEFT": "Move the menu cursor left",
            "RIGHT": "Move the menu cursor right",
            "START": "Open the start menu",
        },
    )
}


def main() -> None:
    client = TypeSafeClient()

    for i in range(4):
        t0 = time.perf_counter()
        result = client.system_one(STATE, QUESTIONS)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        answer = result.choices["next_button"]
        print(
            f"call={i} choice={answer.choice} confidence={answer.confidence:.3f} "
            f"round_trip_ms={elapsed_ms:.1f}"
        )


if __name__ == "__main__":
    main()
