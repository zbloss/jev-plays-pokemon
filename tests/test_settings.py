from pathlib import Path

from jev_plays_pokemon.settings import Settings, load_settings


def test_defaults_to_none_when_nothing_is_set_anywhere(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    settings = load_settings(env_file=tmp_path / "does-not-exist.env")

    assert settings == Settings(
        openai_api_key=None,
        openai_base_url=None,
        openai_vision_model=None,
        dialog_decode_backend=None,
        typesafe_api_key=None,
    )


def test_reads_a_value_from_the_env_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=from-dotenv\n")

    settings = load_settings(env_file=env_file)

    assert settings.openai_api_key == "from-dotenv"


def test_an_environment_variable_beats_the_env_file(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=from-dotenv\n")
    monkeypatch.setenv("OPENAI_API_KEY", "from-envvar")

    settings = load_settings(env_file=env_file)

    assert settings.openai_api_key == "from-envvar"


def test_a_cli_override_beats_the_environment_variable(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    monkeypatch.setenv("OPENAI_API_KEY", "from-envvar")

    settings = load_settings(env_file=env_file, openai_api_key="from-cli")

    assert settings.openai_api_key == "from-cli"


def test_an_unset_cli_override_falls_through_to_the_environment_variable(
    monkeypatch, tmp_path
):
    env_file = tmp_path / ".env"
    monkeypatch.setenv("OPENAI_API_KEY", "from-envvar")

    # `None` means "the flag wasn't passed" - it must not shadow the env var
    # with an explicit `None`.
    settings = load_settings(env_file=env_file, openai_api_key=None)

    assert settings.openai_api_key == "from-envvar"


def test_env_file_accepts_a_path_object(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("TYPESAFE_API_KEY=tsk-test\n")

    settings = load_settings(env_file=Path(env_file))

    assert settings.typesafe_api_key == "tsk-test"
