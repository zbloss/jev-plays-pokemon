"""CLI/`.env`-overridable configuration surface (see docs/adr/0001-...).

Every field defaults to `None`, meaning "not set via a CLI flag, an
environment variable, or `.env`" - callers forward a resolved field straight
into that value's own consumer (`dialog_vision.load_vision_client_and_model`,
`dialog_decode.load_dialog_decoder`, `decision.build_jev_client`), which
already knows what "unconfigured" means for it. This module deliberately
does not invent a second, competing notion of "the default."

Field names double as the env var names pydantic-settings reads
(`openai_api_key` -> `OPENAI_API_KEY`, case-insensitively) - kept identical
to what this project already read via `os.environ.get(...)` before this
module existed, and deliberately unprefixed so `OPENAI_API_KEY`/
`OPENAI_BASE_URL` keep matching the standard `openai` SDK's own
auto-detected env vars (ADR 0001).
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_vision_model: str | None = None
    dialog_decode_backend: str | None = None
    typesafe_api_key: str | None = None


def load_settings(
    env_file: str | Path = ".env", **cli_overrides: str | None
) -> Settings:
    """Resolve `Settings`, giving a non-`None` `cli_overrides` value top priority.

    Precedence: an explicit CLI value > the matching environment variable >
    `env_file` > `None` (unconfigured) - the middle two are pydantic-settings'
    own resolution, applied first; a CLI override is then layered on top via
    `model_copy`. A `None` in `cli_overrides` (the flag wasn't passed) is
    dropped rather than applied, so it falls through to whatever
    environment/`.env` already resolved instead of blanking it out.
    """
    overrides = {
        key: value for key, value in cli_overrides.items() if value is not None
    }
    return Settings(_env_file=env_file).model_copy(update=overrides)
