# jev-plays-pokemon
Like Claude Plays Pokemon, but with Jev

## Usage

Copy `.env.example` to `.env` and fill in at least `TYPESAFE_API_KEY`, then:

```
uv run jev-plays-pokemon              # same as `run`: boot the ROM and play
uv run jev-plays-pokemon run --rom-path pokemon_red.gb --stream-port 8080
uv run jev-plays-pokemon benchmark --turns 25 --warmup 5
uv run jev-plays-pokemon watchdog     # run `run` as a monitored, auto-restarting subprocess
```

Every `.env` value has a matching global CLI flag (e.g. `--openai-api-key`,
`--typesafe-api-key`) that overrides it for a single invocation - see
`uv run jev-plays-pokemon --help` and `docs/adr/0001-unified-typer-cli-and-config-settings.md`.
